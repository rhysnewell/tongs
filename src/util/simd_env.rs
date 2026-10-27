//! Processor-specific floating point setup matching HMMER's `impl_Init()`.

// aarch64 FPCR.FZ flushes denormal inputs and outputs, the same as MXCSR DAZ with FTZ.
pub fn init() {
    unsafe { write_mode(read_mode() | FLUSH_TO_ZERO) }
}

// A library caller's threads keep their own float mode once a search returns.
pub struct FlushToZero {
    saved: usize,
}

impl FlushToZero {
    pub fn enter() -> Self {
        let saved = unsafe { read_mode() };
        init();
        FlushToZero { saved }
    }
}

impl Drop for FlushToZero {
    fn drop(&mut self) {
        unsafe { write_mode(self.saved) }
    }
}

#[cfg(target_arch = "x86_64")]
const FLUSH_TO_ZERO: usize = (1 << 6) | (1 << 15);
#[cfg(target_arch = "aarch64")]
const FLUSH_TO_ZERO: usize = 1 << 24;
#[cfg(not(striped))]
const FLUSH_TO_ZERO: usize = 0;

#[cfg(target_arch = "x86_64")]
unsafe fn read_mode() -> usize {
    let mut csr = 0u32;
    unsafe { std::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack, preserves_flags)) };
    csr as usize
}

#[cfg(target_arch = "x86_64")]
unsafe fn write_mode(mode: usize) {
    let csr = mode as u32;
    unsafe { std::arch::asm!("ldmxcsr [{}]", in(reg) &csr, options(nostack, preserves_flags)) };
}

#[cfg(target_arch = "aarch64")]
unsafe fn read_mode() -> usize {
    let fpcr: usize;
    unsafe {
        std::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags))
    };
    fpcr
}

#[cfg(target_arch = "aarch64")]
unsafe fn write_mode(mode: usize) {
    unsafe {
        std::arch::asm!("msr fpcr, {}", in(reg) mode, options(nomem, nostack, preserves_flags))
    };
}

#[cfg(not(striped))]
unsafe fn read_mode() -> usize {
    0
}

#[cfg(not(striped))]
unsafe fn write_mode(_mode: usize) {}
