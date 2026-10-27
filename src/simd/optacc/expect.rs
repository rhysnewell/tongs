use super::*;

#[derive(Default)]
pub struct Expectation {
    m: Vec<f32>,
    i: Vec<f32>,
    n: f32,
    c: f32,
    j: f32,
}

impl Expectation {
    pub(super) unsafe fn add_row(&mut self, row: usize, cells: &[__m128], px: &[f32]) {
        let q = cells.len() / 3;
        let cells = cells.as_ptr().cast::<f32>();
        let x = &px[row * NXCELLS..row * NXCELLS + NXCELLS];
        if row == 1 {
            self.m.resize(q * 4, 0.0);
            self.i.resize(q * 4, 0.0);
            for qi in 0..q {
                _mm_storeu_ps(self.m.as_mut_ptr().add(qi * 4), load_cell(cells, qi, OM_M));
                _mm_storeu_ps(self.i.as_mut_ptr().add(qi * 4), load_cell(cells, qi, OM_I));
            }
            self.n = x[PXN];
            self.c = x[PXC];
            self.j = x[PXJ];
            return;
        }
        for qi in 0..q {
            let m = self.m.as_mut_ptr().add(qi * 4);
            let i = self.i.as_mut_ptr().add(qi * 4);
            _mm_storeu_ps(m, _mm_add_ps(_mm_loadu_ps(m), load_cell(cells, qi, OM_M)));
            _mm_storeu_ps(i, _mm_add_ps(_mm_loadu_ps(i), load_cell(cells, qi, OM_I)));
        }
        self.n += x[PXN];
        self.c += x[PXC];
        self.j += x[PXJ];
    }

    pub unsafe fn null2(
        &mut self,
        l: usize,
        k: usize,
        rfv: &[Vec<[f32; 4]>],
        null2: &mut Vec<f32>,
    ) {
        if l == 0 {
            return;
        }
        let q = self.m.len() / 4;
        let normv = _mm_set1_ps(1.0 / l as f32);
        for qi in 0..q {
            let m = self.m.as_mut_ptr().add(qi * 4);
            let i = self.i.as_mut_ptr().add(qi * 4);
            _mm_storeu_ps(m, _mm_mul_ps(_mm_loadu_ps(m), normv));
            _mm_storeu_ps(i, _mm_mul_ps(_mm_loadu_ps(i), normv));
        }
        let norm = 1.0 / l as f32;
        let xfactor = self.n * norm + self.c * norm + self.j * norm;
        null2.resize(k, 1.0);
        for (x, odds) in rfv.iter().enumerate().take(k) {
            let mut sv = _mm_setzero_ps();
            for (qi, stripe) in odds.iter().enumerate().take(q) {
                let m = _mm_loadu_ps(self.m.as_ptr().add(qi * 4));
                let i = _mm_loadu_ps(self.i.as_ptr().add(qi * 4));
                sv = _mm_add_ps(sv, _mm_mul_ps(m, _mm_loadu_ps(stripe.as_ptr())));
                sv = _mm_add_ps(sv, i);
            }
            sv = _mm_add_ps(sv, _mm_shuffle_ps::<0b00_11_10_01>(sv, sv));
            sv = _mm_add_ps(sv, _mm_shuffle_ps::<0b01_00_11_10>(sv, sv));
            null2[x] = _mm_cvtss_f32(sv) + xfactor;
        }
    }
}
