mod functions;
mod instant_query;
mod optimizations;
mod range_query;
mod streaming_range_func;

#[cfg(feature = "parquet")]
mod dump_query;
mod inspect_at_modifier;
mod inspect_offset;
#[cfg(feature = "parquet")]
mod parquet_query;
#[cfg(feature = "parquet")]
mod rezolus_query;

/// Register a SIGSEGV handler that prints a backtrace before aborting.
/// This runs before main() via `ctor`.
#[cfg(test)]
#[ctor::ctor]
fn install_sigsegv_handler() {
    unsafe {
        libc::signal(
            libc::SIGSEGV,
            sigsegv_handler as *const () as libc::sighandler_t,
        );
    }
}

#[cfg(test)]
extern "C" fn sigsegv_handler(_sig: libc::c_int) {
    // Use write(2) which is async-signal-safe, unlike eprintln!.
    let msg = b"\n=== SIGSEGV caught ===\n";
    unsafe {
        libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
    }
    // Force capture backtrace (not signal-safe but best-effort).
    let bt = std::backtrace::Backtrace::force_capture();
    let bt_str = format!("{bt}\n");
    let bt_bytes = bt_str.as_bytes();
    unsafe {
        libc::write(2, bt_bytes.as_ptr() as *const libc::c_void, bt_bytes.len());
        libc::signal(libc::SIGSEGV, libc::SIG_DFL);
        libc::raise(libc::SIGSEGV);
    }
}
