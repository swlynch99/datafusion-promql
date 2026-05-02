mod functions;
mod histogram_schema;
mod instant_query;
mod optimizations;
mod range_query;
mod streaming_range_func;

#[cfg(feature = "parquet")]
mod dump_query;
mod inspect_at_modifier;
mod inspect_offset;
#[cfg(feature = "metriken")]
mod metriken;
#[cfg(feature = "parquet")]
mod parquet_query;
#[cfg(feature = "parquet")]
mod rezolus_query;

/// Dummy function whose address we print so we can compute the ASLR base offset.
#[cfg(test)]
#[inline(never)]
fn aslr_anchor() {}

/// Register a SIGSEGV handler on an alternate signal stack so we can capture
/// a backtrace even when the main stack has overflowed.
/// This runs before main() via `ctor`.
#[cfg(test)]
#[ctor::ctor]
fn install_sigsegv_handler() {
    // Print the runtime address of a known symbol so we can compute ASLR offset.
    let anchor_addr = aslr_anchor as *const () as u64;
    eprintln!("ASLR anchor: aslr_anchor runtime={anchor_addr:#018x}");

    unsafe {
        // Allocate a 64 KiB alternate signal stack so the handler can run
        // even when the main stack is exhausted (stack overflow).
        const ALT_STACK_SIZE: usize = 64 * 1024;
        let alt_stack = libc::mmap(
            std::ptr::null_mut(),
            ALT_STACK_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if alt_stack != libc::MAP_FAILED {
            let ss = libc::stack_t {
                ss_sp: alt_stack,
                ss_flags: 0,
                ss_size: ALT_STACK_SIZE,
            };
            libc::sigaltstack(&ss, std::ptr::null_mut());
        }

        // Use sigaction with SA_ONSTACK so the handler runs on the alternate stack.
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sigsegv_handler as *const () as libc::sighandler_t;
        sa.sa_flags = libc::SA_ONSTACK | libc::SA_SIGINFO;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut());
    }
}

#[cfg(test)]
fn write_hex(fd: libc::c_int, val: u64) {
    let mut buf = [b'0'; 18]; // "0x" + 16 hex digits
    buf[0] = b'0';
    buf[1] = b'x';
    let hex = b"0123456789abcdef";
    for i in 0..16 {
        buf[17 - i] = hex[((val >> (i * 4)) & 0xf) as usize];
    }
    unsafe {
        libc::write(fd, buf.as_ptr() as *const libc::c_void, buf.len());
    }
}

#[cfg(test)]
extern "C" fn sigsegv_handler(
    _sig: libc::c_int,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    unsafe {
        let msg = b"\n=== SIGSEGV caught (on alt stack) ===\n";
        libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());

        // Extract faulting address from siginfo_t.
        if !info.is_null() {
            let si = &*info;
            let fault_addr = si.si_addr() as u64;
            let msg = b"  fault addr=";
            libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
            write_hex(2, fault_addr);
            libc::write(2, b"\n".as_ptr() as *const libc::c_void, 1);
        }

        // Extract registers from ucontext_t.
        if !context.is_null() {
            let uc = &*(context as *const libc::ucontext_t);
            let rip = uc.uc_mcontext.gregs[libc::REG_RIP as usize] as u64;
            let rsp = uc.uc_mcontext.gregs[libc::REG_RSP as usize] as u64;
            let rbp = uc.uc_mcontext.gregs[libc::REG_RBP as usize] as u64;

            let msg = b"  rip=";
            libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
            write_hex(2, rip);
            let msg = b" rsp=";
            libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
            write_hex(2, rsp);
            let msg = b" rbp=";
            libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
            write_hex(2, rbp);
            libc::write(2, b"\n".as_ptr() as *const libc::c_void, 1);

            // Dump return addresses by walking the frame pointer chain.
            // We avoid any heap allocation — just print raw addresses.
            let msg = b"  --- return addresses (use addr2line to resolve) ---\n";
            libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());

            write_hex(2, rip);
            libc::write(2, b"\n".as_ptr() as *const libc::c_void, 1);

            let mut frame_bp = rbp as *const u64;
            for _ in 0..200 {
                if frame_bp.is_null() || (frame_bp as u64) < 0x10000 {
                    break;
                }
                if !(frame_bp as usize).is_multiple_of(8) {
                    break;
                }
                // Carefully read from the stack — might fault again.
                // Reset handler to default first so a nested fault terminates.
                libc::signal(libc::SIGSEGV, libc::SIG_DFL);
                let ret_addr = *frame_bp.add(1);
                // Restore our handler for subsequent frames.
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = sigsegv_handler as *const () as libc::sighandler_t;
                sa.sa_flags = libc::SA_ONSTACK | libc::SA_SIGINFO;
                libc::sigemptyset(&mut sa.sa_mask);
                libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut());

                if ret_addr == 0 {
                    break;
                }
                write_hex(2, ret_addr);
                libc::write(2, b"\n".as_ptr() as *const libc::c_void, 1);

                let prev_bp = *frame_bp;
                frame_bp = prev_bp as *const u64;
            }
        }

        libc::signal(libc::SIGSEGV, libc::SIG_DFL);
        libc::raise(libc::SIGSEGV);
    }
}
