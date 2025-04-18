use crate::{ASCII_TABLE_SIZE, pack_rgb, unpack_rgb};

#[inline]
fn timer(label: &str) -> scope_timer::ScopeTimer {
    scope_timer::ScopeTimer::new(label, scope_timer::TimeFormat::Milliseconds, None, false)
}

// 双三次插值权重函数
fn cubic_weight(x: f32, a: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x <= 1.0 {
        (a + 2.0) * abs_x.powi(3) - (a + 3.0) * abs_x.powi(2) + 1.0
    } else if abs_x < 2.0 {
        a * abs_x.powi(3) - 5.0 * a * abs_x.powi(2) + 8.0 * a * abs_x - 4.0 * a
    } else {
        0.0
    }
}

pub fn bicubic_resize(
    dst: &mut [u32],
    dst_width: u32,
    dst_height: u32,
    src: &[u32],
    src_width: u32,
    src_height: u32,
) {
    // 验证输入参数有效性
    assert!(dst.len() >= (dst_width * dst_height) as usize);
    assert!(src.len() >= (src_width * src_height) as usize);

    // 计算缩放比例
    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    // 遍历目标图像每个像素
    for y in 0..dst_height {
        for x in 0..dst_width {
            // 计算对应的源图像位置
            let src_x = x as f32 * x_ratio;
            let src_y = y as f32 * y_ratio;

            let x_floor = src_x.floor() as i32;
            let y_floor = src_y.floor() as i32;

            // 计算16个相邻像素的权重
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut total_weight = 0.0;

            for i in -1..3 {
                for j in -1..3 {
                    let px = (x_floor + i).clamp(0, src_width as i32 - 1) as u32;
                    let py = (y_floor + j).clamp(0, src_height as i32 - 1) as u32;

                    let idx = (py * src_width + px) as usize;
                    let weight_x = cubic_weight(src_x - (x_floor + i) as f32, -0.5);
                    let weight_y = cubic_weight(src_y - (y_floor + j) as f32, -0.5);
                    let weight = weight_x * weight_y;
                    let (sr, sg, sb) = unpack_rgb(src[idx]);

                    r += sr as f32 * weight;
                    g += sg as f32 * weight;
                    b += sb as f32 * weight;
                    total_weight += weight;
                }
            }

            // 归一化并写入目标图像
            let dst_idx = (y * dst_width + x) as usize;
            dst[dst_idx] = pack_rgb(
                (r / total_weight).clamp(0.0, 255.0) as u8,
                (g / total_weight).clamp(0.0, 255.0) as u8,
                (b / total_weight).clamp(0.0, 255.0) as u8,
            );
        }
    }
}

fn repack(
    dst: &mut [u32],
    src: &[u32],
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
) {
    for h in 0..term_height {
        for w in 0..term_width {
            for ch in 0..char_height {
                for cw in 0..char_width {
                    let gh = h * char_height + ch;
                    let gw = w * char_width + cw;
                    let idx_in = (gh * (term_width * char_width) + gw) as usize;
                    let idx_out = (h * term_width * char_height * char_width
                        + w * char_height * char_width
                        + ch * char_width
                        + cw) as usize;
                    dst[idx_out] = src[idx_in];
                }
            }
        }
    }
}

fn compute_brightness(dst: &mut [u32], src: &[u32]) {
    std::iter::zip(dst.iter_mut(), src.iter()).for_each(|(dst, src)| {
        let (r, g, b) = unpack_rgb(*src);
        let brightness = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
        *dst = brightness as u32;
    });
}

fn bimodal_luma_cluster(
    dst: &mut [u32],
    palette: &mut [u32],
    src: &[u32],
    brightness: &[u32],
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
) {
    let px_per_img = (char_width * char_height) as usize;
    let image_count = (term_width * term_height) as usize;

    assert_eq!(palette.len(), image_count * 2);
    assert_eq!(dst.len(), image_count * 4);

    for h in 0..term_height {
        for w in 0..term_width {
            let idx = (h * term_width + w) as usize;
            let img = &src[idx * px_per_img..(idx + 1) * px_per_img];
            let out = &mut dst[idx * 4..(idx + 1) * 4];
            let luma = &brightness[idx * px_per_img..(idx + 1) * px_per_img];

            let (mut l, mut r) = (0, 256);
            while l < r {
                let m = l + (r - l) / 2;
                let mut count = 0;
                for brightness in luma {
                    if *brightness <= m as u32 {
                        count += 1;
                    }
                }
                if count <= luma.len() / 2 {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
            let median = l as u32;

            // 计算两类颜色的平均值
            let (mut sum1, mut cnt1, mut sum2, mut cnt2) = ((0, 0, 0), 0, (0, 0, 0), 0);
            img.iter().zip(luma.iter()).for_each(|(px, &l)| {
                let (r, g, b) = unpack_rgb(*px);
                if l <= median {
                    sum1.0 += r as u32;
                    sum1.1 += g as u32;
                    sum1.2 += b as u32;
                    cnt1 += 1;
                } else {
                    sum2.0 += r as u32;
                    sum2.1 += g as u32;
                    sum2.2 += b as u32;
                    cnt2 += 1;
                }
            });

            match (cnt1, cnt2) {
                (0, 0) => unreachable!(),
                (0, cnt2) => {
                    sum1 = sum2;
                    cnt1 = cnt2;
                }
                (cnt1, 0) => {
                    sum2 = sum1;
                    cnt2 = cnt1;
                }
                _ => {}
            };

            sum2.0 /= cnt2;
            sum2.1 /= cnt2;
            sum2.2 /= cnt2;
            sum1.0 /= cnt1;
            sum1.1 /= cnt1;
            sum1.2 /= cnt1;

            // let mut tmp = Vec::new();
            // for i in 0..4 {
            //     let mut bits = 0u32;
            //     for j in 0..32 {
            //         let bit = if luma[i * 32 + j] <= median { 0 } else { 1 };
            //         bits |= bit << j;
            //         tmp.push(bit);
            //     }
            //     bitset.push(bits);
            // }

            // let mut idx = 0;
            // for i in 0..4 {
            //     for j in 0..32 {
            //         let bit = (bitset[i] >> j) & 0x1;
            //         out[idx] = bit as u8;
            //         out[idx] = tmp[idx] as u8;
            //         idx += 1;
            //     }
            // }

            for i in 0..4 {
                let mut bits = 0u32;
                for j in 0..32 {
                    let bit = if luma[i * 32 + j] <= median { 0 } else { 1 };
                    bits |= bit << j;
                }
                out[i] = bits;
            }

            palette[idx * 2] = pack_rgb(sum1.0 as u8, sum1.1 as u8, sum1.2 as u8);
            palette[idx * 2 + 1] = pack_rgb(sum2.0 as u8, sum2.1 as u8, sum2.2 as u8);
        }
    }
}

fn calc_similarity(
    dst: &mut [u32],
    src: &[u32],
    glyph: &[u32],
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
) {
    for i in 0..term_width * term_height {
        for j in 0..ASCII_TABLE_SIZE * 2 {
            let mut count = 0;
            for k in 0..4 {
                count += (src[(i * 4 + k) as usize] ^ glyph[(j * 4 + k) as usize]).count_zeros();
            }
            dst[(i * ASCII_TABLE_SIZE * 2 + j) as usize] = count;
        }
    }
}

fn get_ascii_string(dst: &mut [u32], src: &[u32], term_width: u32, term_height: u32) {
    for y in 0..term_height {
        for x in 0..term_width {
            let idx = (y * term_width + x) as usize;
            let base = (idx as u32 * ASCII_TABLE_SIZE * 2) as usize;
            let mut max_similarity = 0;
            let mut max_idx = 0;
            for k in 0..ASCII_TABLE_SIZE * 2 {
                let k = k as usize;
                if src[base + k] > max_similarity {
                    max_similarity = src[base + k];
                    max_idx = k;
                }
            }
            let max_idx = max_idx as u32;
            dst[idx] = max_idx;
        }
    }
}
