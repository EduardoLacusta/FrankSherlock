use std::path::PathBuf;

/// Build a `Command` that suppresses console-window creation on Windows.
///
/// On Windows, GUI applications that spawn child processes via `Command::new`
/// cause a visible console window to flash. The `CREATE_NO_WINDOW` flag
/// prevents this. On Linux/macOS this is a no-op wrapper around `Command::new`.
pub fn silent_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    #[allow(unused_mut)]
    let mut cmd = std::process::Command::new(program);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

/// Run a command with a hard time limit.
///
/// Returns `Ok(None)` when the child is still running at the deadline; the
/// child is killed before returning. stdout/stderr are drained on background
/// threads so a chatty child can never fill a pipe buffer and deadlock.
///
/// `Command::output()` waits forever, which is how a stuck helper process
/// (Surya OCR) froze an entire scan with no error in the log.
pub fn run_with_timeout(
    cmd: &mut std::process::Command,
    timeout: std::time::Duration,
) -> std::io::Result<Option<std::process::Output>> {
    use std::io::Read;
    use std::process::Stdio;

    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> std::thread::JoinHandle<Vec<u8>> {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut buf);
            }
            buf
        })
    }
    let out_handle = drain(child.stdout.take());
    let err_handle = drain(child.stderr.take());

    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait()? {
            Some(status) => break Some(status),
            None => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    };

    let Some(status) = status else {
        // Timed out: return immediately. The reader threads are left to finish
        // on their own - a grandchild can keep the pipe open after the child is
        // killed, and waiting on that would defeat the whole point of a timeout.
        return Ok(None);
    };

    let stdout = out_handle.join().unwrap_or_default();
    let stderr = err_handle.join().unwrap_or_default();

    Ok(Some(std::process::Output {
        status,
        stdout,
        stderr,
    }))
}

/// Find an executable by name on the system PATH.
///
/// Uses the `which` crate for cross-platform lookup
/// (handles PATHEXT on Windows automatically).
#[allow(dead_code)]
pub fn find_executable(name: &str) -> Option<PathBuf> {
    which::which(name).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_command() -> std::process::Command {
        #[cfg(target_os = "windows")]
        {
            let mut c = super::silent_command("cmd");
            c.args(["/C", "echo hi"]);
            c
        }
        #[cfg(not(target_os = "windows"))]
        {
            let mut c = super::silent_command("sh");
            c.args(["-c", "echo hi"]);
            c
        }
    }

    fn sleep_command() -> std::process::Command {
        #[cfg(target_os = "windows")]
        {
            let mut c = super::silent_command("cmd");
            // ping is the portable "sleep" on Windows: ~30s
            c.args(["/C", "ping -n 30 127.0.0.1 > nul"]);
            c
        }
        #[cfg(not(target_os = "windows"))]
        {
            let mut c = super::silent_command("sh");
            c.args(["-c", "sleep 30"]);
            c
        }
    }

    #[test]
    fn run_with_timeout_returns_output_for_fast_command() {
        let mut cmd = echo_command();
        let out = run_with_timeout(&mut cmd, std::time::Duration::from_secs(30))
            .unwrap()
            .expect("command should finish before the deadline");
        assert!(out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("hi"));
    }

    #[test]
    fn run_with_timeout_kills_slow_command() {
        let started = std::time::Instant::now();
        let mut cmd = sleep_command();
        let out = run_with_timeout(&mut cmd, std::time::Duration::from_millis(500)).unwrap();
        assert!(out.is_none(), "slow command should hit the deadline");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "should return at the deadline, not wait for the child"
        );
    }

    #[test]
    fn find_executable_known() {
        // Every OS has some basic executable we can test with
        #[cfg(target_os = "windows")]
        let name = "cmd";
        #[cfg(not(target_os = "windows"))]
        let name = "sh";

        let result = find_executable(name);
        assert!(result.is_some(), "should find '{name}' on PATH");
    }

    #[test]
    fn find_executable_nonexistent() {
        let result = find_executable("this_executable_does_not_exist_xyz123");
        assert!(result.is_none());
    }

    #[test]
    fn silent_command_creates_valid_command() {
        #[cfg(target_os = "windows")]
        let program = "cmd";
        #[cfg(not(target_os = "windows"))]
        let program = "echo";

        let cmd = silent_command(program);
        // Just verify it returns a valid Command that can be configured
        let _ = cmd;
    }
}
