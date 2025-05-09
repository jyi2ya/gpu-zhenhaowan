use std::rc::Rc;

use compio::BufResult;
use compio::io::AsyncReadExt;
use compio::io::AsyncWriteExt as _;
use cubecl::Runtime;
use cubecl::prelude::*;
use fontdue::{Font, FontSettings};

const ASCII_START: u32 = 32;
const ASCII_END: u32 = 126;
const ASCII_TABLE_SIZE: u32 = ASCII_END - ASCII_START + 1;

const PACK_R_OFFSET: u32 = 0;
const PACK_G_OFFSET: u32 = 8;
const PACK_B_OFFSET: u32 = 16;

#[no_implicit_prelude]
mod gpu {
    extern crate cubecl;
    extern crate std;

    use cubecl::prelude::*;
    use std::clone::Clone;
    use std::convert::Into;
    use std::default::Default;

    use crate::{ASCII_TABLE_SIZE, PACK_B_OFFSET, PACK_G_OFFSET, PACK_R_OFFSET};

    #[cube(launch)]
    pub fn overlay_edge(image: &mut Array<u32>, edge: &Array<u32>, len: u32) {
        let pos = ABSOLUTE_POS;
        if pos >= len {
            terminate!();
        }

        if edge[pos] == 255 {
            image[pos] = 0;
        }
    }

    #[cube]
    fn pack_rgb(r: u32, g: u32, b: u32) -> u32 {
        (b << PACK_B_OFFSET) | (g << PACK_G_OFFSET) | (r << PACK_R_OFFSET)
    }

    #[cube]
    fn unpack_rgb(packed: u32) -> (u32, u32, u32) {
        let b = (packed >> PACK_B_OFFSET) & 0xff;
        let g = (packed >> PACK_G_OFFSET) & 0xff;
        let r = (packed >> PACK_R_OFFSET) & 0xff;
        (r, g, b)
    }

    #[cube]
    fn cubic_weight(x: f32, a: f32) -> f32 {
        let abs_x = f32::abs(x);
        if abs_x <= 1.0 {
            (a + 2.0) * abs_x * abs_x * abs_x - (a + 3.0) * abs_x * abs_x + 1.0
        } else if abs_x < 2.0 {
            a * abs_x * abs_x * abs_x - 5.0 * a * abs_x * abs_x + 8.0 * a * abs_x - 4.0 * a
        } else {
            f32::new(0.0)
        }
    }

    #[cube(launch)]
    pub fn bicubic_resize(
        dst: &mut Array<u32>,
        src: &Array<u32>,
        dst_width: u32,
        dst_height: u32,
        src_width: u32,
        src_height: u32,
    ) {
        let y = ABSOLUTE_POS_Y;
        let x = ABSOLUTE_POS_X;
        if y >= dst_height || x >= dst_width {
            terminate!();
        }

        let x_ratio = (src_width as f32) / (dst_width as f32);
        let y_ratio = (src_height as f32) / (dst_height as f32);

        let src_x = (x as f32) * x_ratio;
        let src_y = (y as f32) * y_ratio;

        let x_floor = f32::floor(src_x) as i32;
        let y_floor = f32::floor(src_y) as i32;

        // 计算16个相邻像素的权重
        let mut r = 0.0;
        let mut g = 0.0;
        let mut b = 0.0;
        let mut total_weight = 0.0;

        for i in -1..3 {
            for j in -1..3 {
                let px = i32::clamp(x_floor + i, 0, (src_width as i32) - 1) as u32;
                let py = i32::clamp(y_floor + j, 0, (src_height as i32) - 1) as u32;

                let idx = py * src_width + px;
                let weight_x = cubic_weight(src_x - ((x_floor + i) as f32), -0.5);
                let weight_y = cubic_weight(src_y - ((y_floor + j) as f32), -0.5);
                let weight = weight_x * weight_y;
                let (sr, sg, sb) = unpack_rgb(src[idx]);

                r += (sr as f32) * weight;
                g += (sg as f32) * weight;
                b += (sb as f32) * weight;
                total_weight += weight;
            }
        }

        // 归一化并写入目标图像
        let dst_idx = y * dst_width + x;
        dst[dst_idx] = pack_rgb(
            f32::clamp(r / total_weight, 0.0, 255.0) as u32,
            f32::clamp(g / total_weight, 0.0, 255.0) as u32,
            f32::clamp(b / total_weight, 0.0, 255.0) as u32,
        );
    }

    #[cube(launch)]
    pub fn repack(
        dst: &mut Array<u32>,
        src: &Array<u32>,
        term_width: u32,
        term_height: u32,
        char_width: u32,
        char_height: u32,
    ) {
        let gh = ABSOLUTE_POS_Y;
        let gw = ABSOLUTE_POS_X;
        if gh >= term_height * char_height || gw >= term_width * char_width {
            terminate!();
        }
        let h = gh / char_height;
        let ch = gh % char_height;
        let w = gw / char_width;
        let cw = gw % char_width;
        let idx_in = gh * (term_width * char_width) + gw;
        let idx_out = h * term_width * char_height * char_width
            + w * char_height * char_width
            + ch * char_width
            + cw;
        dst[idx_out] = src[idx_in];
    }

    #[cube(launch)]
    pub fn bimodal_luma_cluster(
        dst: &mut Array<u32>,
        palette: &mut Array<u32>,
        src: &Array<u32>,
        brightness: &Array<u32>,
        term_width: u32,
        term_height: u32,
        char_width: u32,
        char_height: u32,
    ) {
        let px_per_img = char_width * char_height;

        let h = ABSOLUTE_POS_Y;
        let w = ABSOLUTE_POS_X;
        if h >= term_height || w >= term_width {
            terminate!();
        }

        let idx = h * term_width + w;

        let mut l = 0;
        let mut r = 256;
        while l < r {
            let m = l + (r - l) / 2;
            let mut count = 0;
            for i in idx * px_per_img..(idx + 1) * px_per_img {
                if brightness[i] <= m {
                    count += 1;
                }
            }
            if count <= px_per_img / 2 {
                l = m + 1;
            } else {
                r = m;
            }
        }
        let median = l;

        let mut sum1_0 = 0;
        let mut sum1_1 = 0;
        let mut sum1_2 = 0;
        let mut sum2_0 = 0;
        let mut sum2_1 = 0;
        let mut sum2_2 = 0;
        let mut cnt1 = 0;
        let mut cnt2 = 0;

        // 计算两类颜色的平均值
        for i in idx * px_per_img..(idx + 1) * px_per_img {
            let (r, g, b) = unpack_rgb(src[i]);
            if brightness[i] <= median {
                sum1_0 += r;
                sum1_1 += g;
                sum1_2 += b;
                cnt1 += 1;
            } else {
                sum2_0 += r;
                sum2_1 += g;
                sum2_2 += b;
                cnt2 += 1;
            }
        }

        if cnt1 == 0 {
            sum1_0 = sum2_0;
            sum1_1 = sum2_1;
            sum1_2 = sum2_2;
            cnt1 = cnt2;
        }

        if cnt2 == 0 {
            sum2_0 = sum1_0;
            sum2_1 = sum1_1;
            sum2_2 = sum1_2;
            cnt2 = cnt1;
        }

        sum2_0 /= cnt2;
        sum2_1 /= cnt2;
        sum2_2 /= cnt2;
        sum1_0 /= cnt1;
        sum1_1 /= cnt1;
        sum1_2 /= cnt1;

        for i in 0..4 {
            let mut bits = 0;
            for j in 0..32 {
                let bit = if brightness[idx * px_per_img + i * 32 + j] <= median {
                    u32::new(0)
                } else {
                    u32::new(1)
                };
                bits |= bit << j;
            }
            dst[idx * 4 + i] = bits;
        }

        // sum1: 暗色部 sum2: 亮色部
        palette[idx * 2] = pack_rgb(sum1_0, sum1_1, sum1_2);
        palette[idx * 2 + 1] = pack_rgb(sum2_0, sum2_1, sum2_2);
    }

    #[cube(launch)]
    pub fn compute_brightness(dst: &mut Array<u32>, src: &Array<u32>, width: u32, height: u32) {
        let y = ABSOLUTE_POS_Y;
        let x = ABSOLUTE_POS_X;
        if y >= height || x >= width {
            terminate!();
        }

        let idx = y * width + x;
        let (r, g, b) = unpack_rgb(src[idx]);
        let brightness = 0.299 * (r as f32) + 0.587 * (g as f32) + 0.114 * (b as f32);
        dst[idx] = brightness as u32;
    }

    #[cube]
    fn count_zeros(x: u32) -> u32 {
        let x = x - ((x >> 1) & 0x55555555);
        let x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        let x = (x + (x >> 4)) & 0x0f0f0f0f;
        let x = x + (x >> 8);
        let x = x + (x >> 16);
        let ones = x & 0x0000003f;

        32 - ones
    }

    #[cube(launch)]
    pub fn calc_similarity(
        dst: &mut Array<u32>,
        src: &Array<u32>,
        glyph: &Array<u32>,
        term_width: u32,
        term_height: u32,
    ) {
        let i = ABSOLUTE_POS_X;
        let j = ABSOLUTE_POS_Y;
        if i >= term_width * term_height || j >= ASCII_TABLE_SIZE * 2 {
            terminate!();
        }

        let mut count = 0;
        for k in 0..4 {
            count += count_zeros(src[i * 4 + k] ^ glyph[j * 4 + k]);
        }
        dst[i * ASCII_TABLE_SIZE * 2 + j] = count;
    }

    #[cube(launch)]
    pub fn get_ascii_string(
        dst: &mut Array<u32>,
        src: &Array<u32>,
        term_width: u32,
        term_height: u32,
    ) {
        let y = ABSOLUTE_POS_Y;
        let x = ABSOLUTE_POS_X;
        if y >= term_height || x >= term_width {
            terminate!();
        }

        let idx = y * term_width + x;
        let base = idx * ASCII_TABLE_SIZE * 2;
        let mut max_similarity = 0;
        let mut max_idx = 0;
        for k in 0..ASCII_TABLE_SIZE * 2 {
            if src[base + k] > max_similarity {
                max_similarity = src[base + k];
                max_idx = k;
            }
        }
        dst[idx] = max_idx;
    }
}

#[allow(dead_code, unused_variables)]
mod cpu;

fn pack_rgb(r: u8, g: u8, b: u8) -> u32 {
    let r = r as u32;
    let g = g as u32;
    let b = b as u32;
    (b << PACK_B_OFFSET) | (g << PACK_G_OFFSET) | (r << PACK_R_OFFSET)
}

fn unpack_rgb(packed: u32) -> (u8, u8, u8) {
    let b = (packed >> PACK_B_OFFSET) & 0xff;
    let g = (packed >> PACK_G_OFFSET) & 0xff;
    let r = (packed >> PACK_R_OFFSET) & 0xff;
    (r as u8, g as u8, b as u8)
}

fn render_centered(font: &Font, character: char) -> [[u8; 8]; 16] {
    // 渲染字符获取位图
    let (metrics, bitmap) = font.rasterize(character, 16.0); // 16px大小

    // 创建8x16矩阵
    let mut matrix = [[0u8; 8]; 16];

    // 计算居中偏移量
    let x_offset = (8 - metrics.width) / 2;
    let y_offset = (16 - metrics.height) / 2;

    // 将位图复制到矩阵中心
    for y in 0..metrics.height {
        for x in 0..metrics.width {
            if y + y_offset < 16 && x + x_offset < 8 {
                matrix[y + y_offset][x + x_offset] = if bitmap[y * metrics.width + x] < 128 {
                    0
                } else {
                    1
                };
            }
        }
    }

    matrix
}

fn generate_ascii_bitmap(font_path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // 加载字体文件
    let font_data = std::fs::read(font_path)?;
    let font = Font::from_bytes(font_data, FontSettings::default())?;

    let mut result = Vec::new();
    // 处理可打印ASCII字符(32-126)
    for c in ASCII_START..=ASCII_END {
        let character = c as u8 as char;

        let bitmap = render_centered(&font, character)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut compressed = Vec::new();
        for i in 0..4 {
            let mut bits = 0;
            for j in 0..32 {
                let bit = bitmap[i * 32 + j] as u32;
                bits |= bit << j;
            }
            compressed.push(bits);
        }
        result.extend(compressed.iter());
        result.extend(compressed.iter().map(|x| !x));
    }

    Ok(result)
}

struct VideoFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

struct VideoStream {
    ffmpeg_stdout: compio::process::ChildStdout,
    width: u32,
    height: u32,
}

impl VideoStream {
    pub fn open(path: &str, width: u32, height: u32) -> Option<Self> {
        let mut process = compio::process::Command::new("ffmpeg");
        process
            .args(["-re"])
            .args(["-loglevel", "error"])
            .args(["-hwaccel", "vaapi"])
            .args(["-hwaccel_output_format", "vaapi"])
            .args(["-i", path])
            .args([
                "-vf",
                format!("scale_vaapi=w={width}:h={height}:format=nv12,hwdownload,format=rgba")
                    .as_str(),
            ])
            // .args([
            //     "-vf",
            //     format!("scale=w={width}:h={height},format=rgba").as_str(),
            // ])
            .args(["-f", "rawvideo"])
            .args(["-pix_fmt", "rgba"])
            .args(["pipe:"])
            .stdout(std::process::Stdio::piped())
            .unwrap();
        let mut process = process.spawn().unwrap();
        let stdout = process.stdout.take().unwrap();

        Some(Self {
            ffmpeg_stdout: stdout,
            width,
            height,
        })
    }

    pub async fn read_next(&mut self) -> Option<VideoFrame> {
        let BufResult(result, data) = self
            .ffmpeg_stdout
            .read_exact(vec![
                0;
                size_of::<u32>() * (self.width * self.height) as usize
            ])
            .await;
        match result {
            Ok(_) => Some(VideoFrame {
                data,
                width: self.width,
                height: self.height,
            }),
            Err(_) => None,
        }
    }
}

fn get_cube_count(shape: [u32; 3], tiling: CubeDim) -> CubeCount {
    CubeCount::new_3d(
        shape[0].div_ceil(tiling.x),
        shape[1].div_ceil(tiling.y),
        shape[2].div_ceil(tiling.z),
    )
}

mod edge {
    use crate::get_cube_count;

    // fn compute_histogram(image: &[u32]) -> [u32; 256] {
    //     let hist = std::array::from_fn(|_| CachePadded::new(AtomicU32::new(0)));
    //     image.par_iter().for_each(|&pixel| {
    //         hist[pixel as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    //     });
    //     hist.map(|x| x.into_inner().into_inner())
    // }

    fn otsu_threshold(hist: &[u32]) -> u8 {
        let total_pixels = hist.iter().sum::<u32>() as f64;
        let mut cumulative_sum = [0u32; 256];
        cumulative_sum[0] = hist[0];
        for i in 1..256 {
            cumulative_sum[i] = cumulative_sum[i - 1] + hist[i];
        }
        let sum_total: u32 = hist.iter().enumerate().map(|(i, &v)| (i as u32) * v).sum();
        let mut max_sigma = 0.0;
        let mut threshold = 0u8;
        for t in 1u8..255 {
            let w0 = (cumulative_sum[usize::from(t)] as f64) / total_pixels;
            let w1 = ((cumulative_sum[255] - cumulative_sum[usize::from(t)]) as f64) / total_pixels;
            if w0 == 0.0 || w1 == 0.0 {
                continue;
            }
            let sum_w0: u32 = hist[..usize::from(t)]
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32) * v)
                .sum();
            let u0 = (sum_w0 as f64) / w0;
            let sum_w1 = sum_total - sum_w0;
            let u1 = (sum_w1 as f64) / w1;
            let sigma = w0 * w1 * (u1 - u0).powi(2);
            if sigma > max_sigma {
                max_sigma = sigma;
                threshold = t;
            }
        }

        // imageproc::edges::canny panics without this
        if threshold == 0 { 1 } else { threshold }
    }

    pub fn otsu_thresholding(hist: &mut [u32]) -> (u8, u8) {
        for i in 1..255 {
            hist[i] = (hist[i - 1] + hist[i] * 2 + hist[i + 1]) / 4;
        }

        let th_otsu = f32::from(otsu_threshold(hist));

        let th1 = th_otsu * 0.7;
        let th2 = th_otsu * 1.1;

        (th1.min(254.0) as u8, th2.min(254.0) as u8)
    }

    // fn guassian_blur(blurred: &mut [u8], image: &[u8], kernel: &[u32], width: u32, height: u32) {
    //     let width = width as usize;
    //     let height = height as usize;
    //     blurred
    //         .par_iter_mut()
    //         .enumerate()
    //         .for_each(|(idx, blurred)| {
    //             let y = idx / width;
    //             let x = idx % width;
    //             if y < 1 || y >= height - 1 || x < 1 || x >= width - 1 {
    //                 return;
    //             }
    //             let mut sum = 0;
    //             for ky in 0..3 {
    //                 for kx in 0..3 {
    //                     let idx = (y + ky - 1) * width + (x + kx - 1);
    //                     sum += image[idx] as usize * kernel[ky * 3 + kx] as usize;
    //                 }
    //             }
    //             *blurred = (sum / 16) as u8;
    //         });
    // }

    // fn compute_gradients(
    //     gradients: &mut [f32],
    //     directions: &mut [f32],
    //     blurred: &[u8],
    //     sobel_x: &[i32],
    //     sobel_y: &[i32],
    //     width: u32,
    //     height: u32,
    // ) {
    //     let width = width as usize;
    //     let height = height as usize;
    //     rayon::iter::IndexedParallelIterator::zip(
    //         gradients.par_iter_mut(),
    //         directions.par_iter_mut(),
    //     )
    //     .enumerate()
    //     .for_each(|(idx, (gradients, directions))| {
    //         let y = idx / width;
    //         let x = idx % width;
    //         if y < 1 || y >= height - 1 || x < 1 || x >= width - 1 {
    //             return;
    //         }
    //         let mut gx = 0;
    //         let mut gy = 0;

    //         for ky in 0..3 {
    //             for kx in 0..3 {
    //                 let idx = (y + ky - 1) * width + (x + kx - 1);
    //                 gx += blurred[idx] as i32 * sobel_x[ky * 3 + kx];
    //                 gy += blurred[idx] as i32 * sobel_y[ky * 3 + kx];
    //             }
    //         }

    //         *gradients = ((gx * gx + gy * gy) as f32).sqrt();
    //         *directions = (gy as f32).atan2(gx as f32);
    //     });
    // }

    // fn non_maximum_suppression(
    //     suppressed: &mut [u8],
    //     gradients: &[f32],
    //     directions: &[f32],
    //     width: u32,
    //     height: u32,
    // ) {
    //     let width = width as usize;
    //     let height = height as usize;
    //     suppressed
    //         .par_iter_mut()
    //         .enumerate()
    //         .for_each(|(idx, suppressed)| {
    //             let y = idx / width;
    //             let x = idx % width;
    //             if y < 1 || y >= height - 1 || x < 1 || x >= width - 1 {
    //                 return;
    //             }

    //             let angle = directions[idx];
    //             let grad = gradients[idx];

    //             // 量化方向到0°,45°,90°,135°
    //             let quantized = if angle < -3.0 * std::f32::consts::PI / 8.0 {
    //                 0
    //             } else if angle < -std::f32::consts::PI / 8.0 {
    //                 1
    //             } else if angle < std::f32::consts::PI / 8.0 {
    //                 0
    //             } else if angle < 3.0 * std::f32::consts::PI / 8.0 {
    //                 3
    //             } else {
    //                 2
    //             };

    //             let (dx1, dy1, dx2, dy2) = match quantized {
    //                 0 => (1, 0, -1, 0),
    //                 1 => (1, 1, -1, -1),
    //                 2 => (0, 1, 0, -1),
    //                 3 => (-1, 1, 1, -1),
    //                 _ => (0, 0, 0, 0),
    //             };

    //             let neighbor1 = gradients[((y as i32 + dy1) * width as i32 + (x as i32 + dx1))
    //                 .clamp(0, i32::MAX) as usize];
    //             let neighbor2 = gradients[((y as i32 + dy2) * width as i32 + (x as i32 + dx2))
    //                 .clamp(0, i32::MAX) as usize];

    //             if grad >= neighbor1 && grad >= neighbor2 {
    //                 *suppressed = grad as u8;
    //             }
    //         });
    // }

    // fn hysteresis_strong(edges: &mut [u8], suppressed: &[u8], width: u32, height: u32, high: u32) {
    //     let width = width as usize;
    //     let height = height as usize;
    //     let high = high as u8;

    //     rayon::iter::IndexedParallelIterator::zip(edges.par_iter_mut(), suppressed.par_iter())
    //         .enumerate()
    //         .for_each(|(idx, (edges, suppressed))| {
    //             let y = idx / width;
    //             let x = idx % width;
    //             if y < 1 || y >= height - 1 || x < 1 || x >= width - 1 {
    //                 return;
    //             }
    //             if *suppressed >= high {
    //                 *edges = 255;
    //             }
    //         });
    // }

    // fn hysteresis_weak(edges: &mut [u8], suppressed: &[u8], width: u32, height: u32, low: u32) {
    //     let low = low as u8;

    //     for y in 1..height - 1 {
    //         for x in 1..width - 1 {
    //             let idx = (y * width + x) as usize;
    //             if suppressed[idx] >= low && edges[idx] == 0 {
    //                 // 检查8邻域是否有强边缘
    //                 for ky in -1..=1 {
    //                     for kx in -1..=1 {
    //                         if ky == 0 && kx == 0 {
    //                             continue;
    //                         }
    //                         let nidx = (y as i32 + ky) as usize * width as usize
    //                             + (x as i32 + kx) as usize;
    //                         if edges[nidx] == 255 {
    //                             edges[idx] = 255;
    //                             break;
    //                         }
    //                     }
    //                     if edges[idx] == 255 {
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    #[no_implicit_prelude]
    pub mod gpu {
        extern crate cubecl;
        extern crate std;

        use cubecl::prelude::*;
        use std::clone::Clone;
        use std::convert::Into;
        use std::default::Default;

        #[cube(launch)]
        pub fn compute_histogram(hist: &mut Array<Atomic<u32>>, image: &Array<u32>, len: u32) {
            let pos = ABSOLUTE_POS;
            if pos >= len {
                terminate!();
            }
            Atomic::add(&hist[image[pos]], 1);
        }

        #[cube(launch)]
        pub fn guassian_blur(
            blurred: &mut Array<u32>,
            image: &Array<u32>,
            kernel: &Array<u32>,
            width: u32,
            height: u32,
        ) {
            let y = ABSOLUTE_POS_Y;
            let x = ABSOLUTE_POS_X;
            if y < 1 || y >= height - 1 || x < 1 || x >= width - 1 {
                terminate!();
            }
            let idx = y * width + x;
            let mut sum = 0;
            for ky in 0..3 {
                for kx in 0..3 {
                    let idx = (y + ky - 1) * width + (x + kx - 1);
                    sum += image[idx] * kernel[ky * 3 + kx];
                }
            }
            blurred[idx] = sum / 16;
        }

        #[cube]
        fn atan(x: f32) -> f32 {
            let x2 = x * x;
            let mut sum = x;
            let mut term = x;
            let mut n = u32::new(1);

            // 迭代次数可调
            while n < 5 {
                term = -term * x2;
                sum += term / ((2 * n + 1) as f32);
                n += 1;
            }
            sum
        }

        #[cube]
        fn atan2(y: f32, x: f32) -> f32 {
            if x > 0.0 {
                atan(y / x)
            } else if x < 0.0 {
                let at = atan(y / x);
                if y >= 0.0 {
                    at + std::f32::consts::PI
                } else {
                    at - std::f32::consts::PI
                }
            } else if y > 0.0 {
                f32::new(std::f32::consts::FRAC_PI_2)
            } else if y < 0.0 {
                -f32::new(std::f32::consts::FRAC_PI_2)
            } else {
                f32::new(0.0)
            }
        }

        #[cube(launch)]
        pub fn compute_gradients(
            gradients: &mut Array<f32>,
            directions: &mut Array<f32>,
            blurred: &Array<u32>,
            sobel_x: &Array<i32>,
            sobel_y: &Array<i32>,
            width: u32,
            height: u32,
        ) {
            let y = ABSOLUTE_POS_Y;
            let x = ABSOLUTE_POS_X;
            if y < 1 || y >= height - 1 || x < 1 || x >= width - 1 {
                terminate!();
            }
            let idx = y * width + x;
            let mut gx = 0;
            let mut gy = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let idx = (y + ky - 1) * width + (x + kx - 1);
                    gx += (blurred[idx] as i32) * sobel_x[ky * 3 + kx];
                    gy += (blurred[idx] as i32) * sobel_y[ky * 3 + kx];
                }
            }

            gradients[idx] = f32::sqrt((gx * gx + gy * gy) as f32);
            directions[idx] = atan2(gy as f32, gx as f32);
        }

        #[allow(unused_assignments)]
        #[cube(launch)]
        pub fn non_maximum_suppression(
            suppressed: &mut Array<u32>,
            gradients: &Array<f32>,
            directions: &Array<f32>,
            width: u32,
            height: u32,
        ) {
            let y = ABSOLUTE_POS_Y;
            let x = ABSOLUTE_POS_X;
            if y < 1 || y >= height - 1 || x < 1 || x >= width - 1 {
                terminate!();
            }
            let idx = y * width + x;

            let angle = directions[idx];
            let grad = gradients[idx];

            let mut dx1 = 0i32;
            let mut dy1 = 0i32;
            let mut dx2 = 0i32;
            let mut dy2 = 0i32;

            // 量化方向到0°,45°,90°,135°
            if angle < (-3.0 * std::f32::consts::PI) / 8.0 {
                dx1 = 1;
                dy1 = 0;
                dx2 = -1;
                dy2 = 0;
            } else if angle < -std::f32::consts::PI / 8.0 {
                dx1 = 1;
                dy1 = 1;
                dx2 = -1;
                dy2 = -1;
            } else if angle < std::f32::consts::PI / 8.0 {
                dx1 = 1;
                dy1 = 0;
                dx2 = -1;
                dy2 = 0;
            } else if angle < (3.0 * std::f32::consts::PI) / 8.0 {
                dx1 = -1;
                dy1 = 1;
                dx2 = 1;
                dy2 = -1;
            } else {
                dx1 = 0;
                dy1 = 1;
                dx2 = 0;
                dy2 = -1;
            }

            let neighbor1 = gradients
                [i32::max(((y as i32) + dy1) * (width as i32) + ((x as i32) + dx1), 0) as u32];
            let neighbor2 = gradients
                [i32::max(((y as i32) + dy2) * (width as i32) + ((x as i32) + dx2), 0) as u32];

            if grad >= neighbor1 && grad >= neighbor2 {
                suppressed[idx] = grad as u32;
            } else {
                suppressed[idx] = 0;
            }
        }

        #[cube(launch)]
        pub fn hysteresis_strong(
            edges: &mut Array<u32>,
            suppressed: &Array<u32>,
            len: u32,
            high: u32,
        ) {
            let idx = ABSOLUTE_POS;
            if idx >= len {
                terminate!();
            }
            if suppressed[idx] >= high {
                edges[idx] = 255;
            } else {
                edges[idx] = 0;
            }
        }

        #[cube(launch)]
        pub fn hysteresis_weak(
            edges: &mut Array<u32>,
            suppressed: &Array<u32>,
            width: u32,
            height: u32,
            low: u32,
        ) {
            let y = ABSOLUTE_POS_Y;
            let x = ABSOLUTE_POS_X;
            if y < 1 || y >= height - 1 || x < 1 || x >= width - 1 {
                terminate!();
            }
            let idx = y * width + x;
            if suppressed[idx] >= low && edges[idx] == 0 {
                // 检查8邻域是否有强边缘
                for ky in -1..=1 {
                    for kx in -1..=1 {
                        if ky != 0 || kx != 0 {
                            let nidx = ((y as i32) + ky) * (width as i32) + ((x as i32) + kx);
                            if edges[nidx as u32] == 255 {
                                edges[idx] = 255;
                                break;
                            }
                        }
                    }
                    if edges[idx] == 255 {
                        break;
                    }
                }
            }
        }
    }

    pub fn canny<RT: cubecl::Runtime>(
        client: &cubecl::client::ComputeClient<RT::Server, RT::Channel>,
        image: cubecl::server::Handle,
        width: u32,
        height: u32,
        low_threshold: u32,
        high_threshold: u32,
    ) -> cubecl::server::Handle {
        use cubecl::prelude::*;

        let tiling = CubeDim::new_2d(16, 16);

        let len = (width * height) as usize;

        let blurred = client.empty(size_of::<u32>() * len);
        let kernel = client.create(u32::as_bytes(&[1, 2, 1, 2, 4, 2, 1, 2, 1]));
        unsafe {
            gpu::guassian_blur::launch::<RT>(
                client,
                get_cube_count([width, height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(&blurred, len, 1),
                ArrayArg::from_raw_parts::<u32>(&image, len, 1),
                ArrayArg::from_raw_parts::<u32>(&kernel, 9, 1),
                ScalarArg::new(width),
                ScalarArg::new(height),
            );
        }

        // 1. 高斯滤波
        // let mut blurred = vec![0u8; len];
        // let gauss_kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
        // guassian_blur(&mut blurred, &gray_image, &gauss_kernel, width, height);
        // let blurred = blurred.into_iter().map(|x| x as u32).collect::<Vec<_>>();
        // let blurred = client.create(u32::as_bytes(&blurred));

        let gradients = client.empty(size_of::<f32>() * len);
        let directions = client.empty(size_of::<f32>() * len);
        let sobel_x = client.create(i32::as_bytes(&[-1, 0, 1, -2, 0, 2, -1, 0, 1]));
        let sobel_y = client.create(i32::as_bytes(&[-1, -2, -1, 0, 0, 0, 1, 2, 1]));
        unsafe {
            gpu::compute_gradients::launch::<RT>(
                client,
                get_cube_count([width, height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<f32>(&gradients, len, 1),
                ArrayArg::from_raw_parts::<f32>(&directions, len, 1),
                ArrayArg::from_raw_parts::<u32>(&blurred, len, 1),
                ArrayArg::from_raw_parts::<i32>(&sobel_x, 9, 1),
                ArrayArg::from_raw_parts::<i32>(&sobel_y, 9, 1),
                ScalarArg::new(width),
                ScalarArg::new(height),
            );
        }

        // // 2. 计算梯度
        // let mut gradients = vec![0f32; len];
        // let mut directions = vec![0f32; len];
        // let sobel_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        // let sobel_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
        // compute_gradients(
        //     &mut gradients,
        //     &mut directions,
        //     &blurred,
        //     &sobel_x,
        //     &sobel_y,
        //     width,
        //     height,
        // );
        // let gradients = client.create(f32::as_bytes(&gradients));
        // let directions = client.create(f32::as_bytes(&directions));

        // 3. 非极大值抑制
        // let mut suppressed = vec![0u8; len];
        let suppressed = client.empty(size_of::<u32>() * len);
        unsafe {
            gpu::non_maximum_suppression::launch::<RT>(
                client,
                get_cube_count([width, height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(&suppressed, len, 1),
                ArrayArg::from_raw_parts::<u32>(&gradients, len, 1),
                ArrayArg::from_raw_parts::<u32>(&directions, len, 1),
                ScalarArg::new(width),
                ScalarArg::new(height),
            );
        }
        // non_maximum_suppression(&mut suppressed, &gradients, &directions, width, height);

        let edges = client.empty(size_of::<u32>() * len);

        unsafe {
            gpu::hysteresis_strong::launch::<RT>(
                client,
                get_cube_count([len as u32, 1, 1], CubeDim::new(256, 1, 1)),
                CubeDim::new(256, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&edges, len, 1),
                ArrayArg::from_raw_parts::<u32>(&suppressed, len, 1),
                ScalarArg::new(len as u32),
                ScalarArg::new(high_threshold),
            );
        }

        for _ in 0..5 {
            unsafe {
                gpu::hysteresis_weak::launch::<RT>(
                    client,
                    get_cube_count([width, height, 1], tiling),
                    tiling,
                    ArrayArg::from_raw_parts::<u32>(&edges, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&suppressed, len, 1),
                    ScalarArg::new(width),
                    ScalarArg::new(height),
                    ScalarArg::new(low_threshold),
                );
            }
        }

        edges
    }
}

#[allow(clippy::too_many_arguments)]
async fn render<RT: cubecl::prelude::Runtime>(
    client: &cubecl::client::ComputeClient<RT::Server, RT::Channel>,
    data: &[u8],
    width: u32,
    height: u32,
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
    glyph: &cubecl::server::Handle,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let tiling = CubeDim::new_2d(16, 16);

    let resized = client.create(data);
    let (resized_width, resized_height) = (width, height);

    let edges = {
        let single_channel_image =
            client.empty(size_of::<u32>() * ((resized_width * resized_height) as usize));
        unsafe {
            gpu::compute_brightness::launch::<RT>(
                client,
                get_cube_count([resized_width, resized_height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(
                    &single_channel_image,
                    (resized_height * resized_width) as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(
                    &resized,
                    (resized_height * resized_width) as usize,
                    1,
                ),
                ScalarArg::new(resized_width),
                ScalarArg::new(resized_height),
            );
        }
        let hist = client.create(u32::as_bytes(&[0; 256]));
        let len = resized_height * resized_width;
        unsafe {
            edge::gpu::compute_histogram::launch::<RT>(
                client,
                get_cube_count([len, 1, 1], CubeDim::new_1d(256)),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts::<u32>(&hist, len as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&single_channel_image, len as usize, 1),
                ScalarArg::new(len),
            );
        }
        let mut hist = u32::from_bytes(&client.read_one_async(hist.binding()).await).to_owned();
        let (canny_low, canny_high) = edge::otsu_thresholding(&mut hist);
        edge::canny::<RT>(
            client,
            single_channel_image,
            width,
            height,
            canny_low as u32,
            canny_high as u32,
        )
    };

    unsafe {
        let len = resized_width * resized_height;
        gpu::overlay_edge::launch::<RT>(
            client,
            CubeCount::new_1d(len.div_ceil(256)),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts::<u32>(&resized, len as usize, 1),
            ArrayArg::from_raw_parts::<u32>(&edges, len as usize, 1),
            ScalarArg::new(len),
        );
    }

    // let image = u32::from_bytes(data);
    // let enhanced = image
    //     .into_iter()
    //     .enumerate()
    //     .map(|(idx, &img)| {
    //         let offset = idx % 32;
    //         let idx = idx / 32;
    //         let overlay = (edges[idx] >> (31 - offset)) & 0x1;
    //         match overlay {
    //             1 => 0,
    //             _ => img,
    //         }
    //     })
    //     .collect::<Vec<_>>();

    // let (resized, resized_width, resized_height) = {
    //     // let resized = client.create(u32::as_bytes(&enhanced));
    //     let resized_width = width;
    //     let resized_height = height;

    //     // // let _t = timer("resize");
    //     // let resized_width = term_width * char_width;
    //     // let resized_height = term_height * char_height;
    //     // let resized_len = (resized_height * resized_width) as usize;

    //     // let resized = client.empty(size_of::<u32>() * resized_len);
    //     // let input = client.create(u32::as_bytes(data));

    //     // unsafe {
    //     //     gpu::bicubic_resize::launch::<RT>(
    //     //         client,
    //     //         get_cube_count([resized_width, resized_height, 1], tiling),
    //     //         tiling,
    //     //         ArrayArg::from_raw_parts::<u32>(&resized, resized_len, 1),
    //     //         ArrayArg::from_raw_parts::<u32>(&input, data.len(), 1),
    //     //         ScalarArg::new(resized_width),
    //     //         ScalarArg::new(resized_height),
    //     //         ScalarArg::new(width),
    //     //         ScalarArg::new(height),
    //     //     );
    //     // };
    //     (resized, resized_width, resized_height)
    // };

    let (repacked, repacked_width, repacked_height) = {
        // let _t = timer("repack");

        let repacked_width = resized_width;
        let repacked_height = resized_height;
        let repacked =
            client.empty(size_of::<u32>() * ((repacked_width * repacked_height) as usize));
        unsafe {
            gpu::repack::launch::<RT>(
                client,
                get_cube_count([repacked_width, repacked_height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(
                    &repacked,
                    (repacked_width * repacked_height) as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(
                    &resized,
                    (resized_width * resized_height) as usize,
                    1,
                ),
                ScalarArg::new(term_width),
                ScalarArg::new(term_height),
                ScalarArg::new(char_width),
                ScalarArg::new(char_height),
            );
        }

        (repacked, repacked_width, repacked_height)
    };

    let (clustered, palette, clustered_width, clustered_height) = {
        // let _t = timer("cluster");

        let brightness =
            client.empty(size_of::<u32>() * ((repacked_height * repacked_width) as usize));
        unsafe {
            gpu::compute_brightness::launch::<RT>(
                client,
                get_cube_count([repacked_width, repacked_height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(
                    &brightness,
                    (repacked_height * repacked_width) as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(
                    &repacked,
                    (repacked_height * repacked_width) as usize,
                    1,
                ),
                ScalarArg::new(repacked_width),
                ScalarArg::new(repacked_height),
            );
        }

        let clustered_width = resized_width;
        let clustered_height = resized_height;
        let clustered = client.empty(size_of::<u32>() * ((term_width * term_height * 4) as usize));
        let palette = client.empty(size_of::<u32>() * ((term_width * term_height * 2) as usize));
        unsafe {
            gpu::bimodal_luma_cluster::launch::<RT>(
                client,
                get_cube_count([term_width, term_height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(
                    &clustered,
                    (term_width * term_height * 4) as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(
                    &palette,
                    (term_width * term_height * 2) as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(
                    &repacked,
                    (repacked_width * repacked_height) as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(
                    &brightness,
                    (repacked_width * repacked_height) as usize,
                    1,
                ),
                ScalarArg::new(term_width),
                ScalarArg::new(term_height),
                ScalarArg::new(char_width),
                ScalarArg::new(char_height),
            );
        }

        (clustered, palette, clustered_width, clustered_height)
    };

    let similarity = {
        // let _t = timer("similarity");
        let similarity = client
            .empty(size_of::<u32>() * ((term_width * term_height * ASCII_TABLE_SIZE * 2) as usize));
        unsafe {
            gpu::calc_similarity::launch::<RT>(
                client,
                get_cube_count([term_width * term_height, ASCII_TABLE_SIZE * 2, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(
                    &similarity,
                    (term_width * term_height * ASCII_TABLE_SIZE * 2) as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(
                    &clustered,
                    (clustered_width * clustered_height) as usize,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(glyph, (ASCII_TABLE_SIZE * 2) as usize, 1),
                ScalarArg::new(term_width),
                ScalarArg::new(term_height),
            );
        }

        similarity
    };

    let term = {
        let term = client.empty(size_of::<u32>() * ((term_width * term_height) as usize));
        unsafe {
            gpu::get_ascii_string::launch::<RT>(
                client,
                get_cube_count([term_width, term_height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(&term, (term_width * term_height) as usize, 1),
                ArrayArg::from_raw_parts::<u32>(
                    &similarity,
                    (term_width * term_height * ASCII_TABLE_SIZE * 2) as usize,
                    1,
                ),
                ScalarArg::new(term_width),
                ScalarArg::new(term_height),
            );
        }
        term
    };

    let palette = u32::from_bytes(&client.read_one_async(palette.binding()).await).to_owned();
    let term = u32::from_bytes(&client.read_one_async(term.binding()).await).to_owned();
    Ok((term, palette))
}

fn screen(term: Vec<u32>, palette: Vec<u32>, term_width: u32, term_height: u32) -> String {
    let mut output = String::with_capacity((term_width * term_height * 20) as usize);
    let mut idx = 0;

    output.push_str("\x1b[0;0H"); // 移动到(0,0)

    for _line in 0..term_height {
        for _row in 0..term_width {
            let dark = palette[idx * 2];
            let bright = palette[idx * 2 + 1];

            let (fg_color, bg_color) = if term[idx] % 2 == 0 {
                (bright, dark)
            } else {
                (dark, bright)
            };

            let chr = char::from((term[idx] / 2 + ASCII_START) as u8);

            let (fr, fg, fb) = unpack_rgb(fg_color);
            let (br, bg, bb) = unpack_rgb(bg_color);

            output.push_str(
                format!("\x1b[38;2;{fr};{fg};{fb}m\x1b[48;2;{br};{bg};{bb}m{chr}").as_str(),
            );

            idx += 1;
        }
        output.push_str("\x1b[0m\n"); // 重置颜色并换行
    }

    output
}

#[compio::main]
async fn main() -> anyhow::Result<()> {
    let file = std::env::args().nth(1).expect("usage: $0 <video_path>");
    let bitmap_data = generate_ascii_bitmap("font.otf").unwrap();
    println!("Generated {} bytes of bitmap data", bitmap_data.len());

    let (term_width, term_height) = crossterm::terminal::size()?;
    let term_width = (term_width as u32) - 2;
    let term_height = (term_height as u32) - 2;

    let char_width = 8;
    let char_height = 16;

    let (tx, rx) = async_channel::bounded::<compio::runtime::Task<Result<_, _>>>(1);

    compio::runtime
        ::spawn(async move {
            let mut frames_rendered = 0;
            let mut frames_dropped = 0;
            let start = std::time::Instant::now();
            let mut handles = Vec::new();
            while let Ok(finished) = rx.recv().await {
                handles.push(finished);
                if handles.len() > 30 {
                    drop(handles.remove(0));
                    frames_dropped += 1;
                }

                if rx.is_empty() {
                    if
                        let Some(finished) = handles
                            .iter()
                            .enumerate()
                            .find(|(_idx, task)| task.is_finished())
                            .map(|(idx, _task)| idx)
                    {
                        let mut rest = handles.split_off(finished);
                        let finished = rest.remove(0);
                        frames_dropped += handles.len();
                        handles = rest;

                        let (term, palette) = finished.await.unwrap();
                        let now = std::time::Instant::now();
                        let duration = (now - start).as_secs_f64();
                        let screen = screen(term, palette, term_width, term_height);
                        let content = format!(
                            "{screen}ok, {duration:.2} secs, {frames_rendered} frames, {frames_dropped} dropped, {:.2}/{:.2} fps",
                            (frames_rendered as f64) / duration,
                            ((frames_rendered + frames_dropped) as f64) / duration
                        );
                        compio::fs::stdout().write_all(content).await.unwrap();
                        frames_rendered += 1;
                    }
                }
            }
        })
        .detach();

    let mut stream =
        VideoStream::open(&file, term_width * char_width, term_height * char_height).unwrap();
    let client = Rc::new(cubecl::wgpu::WgpuRuntime::client(&Default::default()));
    let glyph = Rc::new(client.create(u32::as_bytes(&bitmap_data)));
    while let Some(frame) = stream.read_next().await {
        let client = Rc::clone(&client);
        let glyph = Rc::clone(&glyph);
        let handle = compio::runtime::spawn(async move {
            render::<cubecl::wgpu::WgpuRuntime>(
                &client,
                &frame.data,
                frame.width,
                frame.height,
                term_width,
                term_height,
                char_width,
                char_height,
                &glyph,
            )
            .await
            .unwrap()
        });
        tx.send(handle).await.unwrap();
    }

    Ok(())
}
