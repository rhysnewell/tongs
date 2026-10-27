//! Processor-specific floating point setup matching HMMER's `impl_Init()`.

pub fn init() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        write_mxcsr(read_mxcsr() | MXCSR_DAZ | MXCSR_FTZ);
    }
}

// A library caller's threads keep their own float mode once a search returns.
pub struct FlushToZero {
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}

impl FlushToZero {
    pub fn enter() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let saved = unsafe { read_mxcsr() };
            init();
            FlushToZero { saved }
        }
        #[cfg(not(target_arch = "x86_64"))]
        FlushToZero {}
    }
}

impl Drop for FlushToZero {
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            write_mxcsr(self.saved);
        }
    }
}

#[cfg(target_arch = "x86_64")]
const MXCSR_DAZ: u32 = 1 << 6;
#[cfg(target_arch = "x86_64")]
const MXCSR_FTZ: u32 = 1 << 15;

#[cfg(target_arch = "x86_64")]
unsafe fn read_mxcsr() -> u32 {
    let mut csr = 0u32;
    unsafe { std::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack, preserves_flags)) };
    csr
}

#[cfg(target_arch = "x86_64")]
unsafe fn write_mxcsr(csr: u32) {
    unsafe { std::arch::asm!("ldmxcsr [{}]", in(reg) &csr, options(nostack, preserves_flags)) };
}
