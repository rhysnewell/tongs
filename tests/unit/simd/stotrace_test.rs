use super::*;

fn exact<const N: usize>(roll: f64, weights: [f32; N]) -> usize {
    let mut probs = weights;
    f_norm(&mut probs);
    pick_discrete(roll, &probs)
}

fn exact_cuts<const N: usize>(weights: [f32; N]) -> Vec<f64> {
    let mut probs = weights;
    f_norm(&mut probs);
    let norm = probs.iter().fold(0.0_f64, |sum, &p| sum + p as f64);
    let mut sum = 0.0_f64;
    probs
        .iter()
        .map(|&p| {
            sum += p as f64;
            sum / norm
        })
        .collect()
}

fn weights<const N: usize>(rng: &mut MersenneTwister) -> [f32; N] {
    let scale = 10.0_f32.powf(-20.0 + 30.0 * rng.next_f64() as f32);
    let top = rng.next_u32() as usize % N;
    std::array::from_fn(|i| match rng.next_u32() % 8 {
        _ if i == top => scale,
        0 => 0.0,
        1 => scale,
        _ => scale * 10.0_f32.powf(-12.0 + 12.0 * rng.next_f64() as f32),
    })
}

fn near(cut: f64) -> impl Iterator<Item = f64> {
    let mut lo = cut;
    let mut hi = cut;
    std::iter::once(cut).chain((0..4).flat_map(move |_| {
        lo = lo.next_down();
        hi = hi.next_up();
        [lo, hi]
    }))
}

fn check<const N: usize>(rng: &mut MersenneTwister, quick: &mut usize, uniform: &mut usize) {
    let w = weights::<N>(rng);
    let boundary = exact_cuts(w).into_iter().flat_map(near);
    for roll in boundary.filter(|r| (0.0..1.0).contains(r)) {
        if let Some(i) = quick_pick(roll, &w) {
            assert_eq!(i, exact(roll, w), "{w:?} roll {roll:e}");
        }
    }
    for _ in 0..16 {
        let roll = rng.next_f64();
        *uniform += 1;
        if let Some(i) = quick_pick(roll, &w) {
            assert_eq!(i, exact(roll, w), "{w:?} roll {roll:e}");
            *quick += 1;
        }
    }
}

#[test]
fn quick_pick_matches_easel_at_every_cut() {
    let mut rng = MersenneTwister::new(42);
    let (mut quick, mut uniform) = (0, 0);
    for _ in 0..50_000 {
        check::<2>(&mut rng, &mut quick, &mut uniform);
        check::<4>(&mut rng, &mut quick, &mut uniform);
    }
    assert!(quick as f64 > 0.99 * uniform as f64, "{quick} of {uniform}");
}
