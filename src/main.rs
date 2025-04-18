use anyhow::Result;
use cubecl::Runtime;
use cubecl::prelude::*;
use fontdue::{Font, FontSettings};
use std::fs;
use tokio::io::AsyncWriteExt;

#[inline]
fn timer(label: &str) -> scope_timer::ScopeTimer {
    scope_timer::ScopeTimer::new(label, scope_timer::TimeFormat::Milliseconds, None, false)
}

fn pack_rgb(r: u8, g: u8, b: u8) -> u32 {
    let r = r as u32;
    let g = g as u32;
    let b = b as u32;
    b << 16 | g << 8 | r
}

fn unpack_rgb(packed: u32) -> (u8, u8, u8) {
    let b = packed >> 16;
    let g = (packed >> 8) & 0xff;
    let r = packed & 0xff;
    (r as u8, g as u8, b as u8)
}

#[cfg(test)]
mod test {
    use crate::unpack_rgb;

    #[test]
    fn test_pack_unpack() {
        let rgb = (12, 33, 86);
        let mem = [rgb.0, rgb.1, rgb.2, 0];
        let mem_packed = u32::from_ne_bytes(mem);
        let packed = super::pack_rgb(rgb.0, rgb.1, rgb.2);
        assert_eq!(packed, mem_packed);
        let unpack = super::unpack_rgb(packed);
        assert_eq!(rgb, unpack);
    }
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

const ASCII_START: u32 = 32;
const ASCII_END: u32 = 126;
const ASCII_TABLE_SIZE: u32 = ASCII_END - ASCII_START + 1;

fn generate_ascii_bitmap(font_path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // 加载字体文件
    let font_data = fs::read(font_path)?;
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

#[no_implicit_prelude]
mod gpu {
    extern crate cubecl;
    extern crate std;

    use cubecl::prelude::*;
    use std::clone::Clone;
    use std::convert::Into;
    use std::default::Default;

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
        dst: &mut Tensor<u8>,
        src: &Tensor<u8>,
        #[comptime] dst_width: u32,
        #[comptime] dst_height: u32,
        #[comptime] src_width: u32,
        #[comptime] src_height: u32,
    ) {
        let y = ABSOLUTE_POS_Y;
        let x = ABSOLUTE_POS_X;
        if y >= dst_height || x >= dst_width {
            terminate!();
        }

        let x_ratio = src_width as f32 / dst_width as f32;
        let y_ratio = src_height as f32 / dst_height as f32;

        let src_x = x as f32 * x_ratio;
        let src_y = y as f32 * y_ratio;

        let x_floor = src_x as i32;
        let y_floor = src_y as i32;

        let mut r = 0.0;
        let mut g = 0.0;
        let mut b = 0.0;
        let mut total_weight = 0.0;

        for i in -1..3 {
            for j in -1..3 {
                let px = i32::clamp(x_floor + i, 0i32, src_width as i32 - 1) as u32;
                let py = i32::clamp(y_floor + j, 0i32, src_height as i32 - 1) as u32;

                let idx = (py * src_width + px) * 4;
                let weight_x = cubic_weight(src_x - (x_floor + i) as f32, -0.5);
                let weight_y = cubic_weight(src_y - (y_floor + j) as f32, -0.5);
                let weight = weight_x * weight_y;

                r += src[idx] as f32 * weight;
                g += src[idx + 1] as f32 * weight;
                b += src[idx + 2] as f32 * weight;
                total_weight += weight;
            }
        }

        let dst_idx = (y * dst_width + x) * 4;
        dst[dst_idx] = f32::clamp(r / total_weight, 0.0, 255.0) as u8;
        dst[dst_idx + 1] = f32::clamp(g / total_weight, 0.0, 255.0) as u8;
        dst[dst_idx + 2] = f32::clamp(b / total_weight, 0.0, 255.0) as u8;
        dst[dst_idx + 3] = 255;
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

fn _unpack(
    dst: &mut [u8],
    src: &[u8],
    palette: &[[[u8; 4]; 2]],
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
                    let idx_in_start = (gh * (term_width * char_width) + gw) as usize * 4;
                    let idx_in_end = idx_in_start + 4;

                    let block_idx = (h * term_width + w) as usize;
                    let idx_out = (h * term_width * char_height * char_width
                        + w * char_height * char_width
                        + ch * char_width
                        + cw) as usize;
                    let color = palette[block_idx][src[idx_out] as usize];

                    (&mut dst[idx_in_start..idx_in_end]).copy_from_slice(&color);
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
    palette: &mut [(u32, u32)],
    src: &[u32],
    brightness: &[u32],
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
) {
    let px_per_img = (char_width * char_height) as usize;
    let image_count = (term_width * term_height) as usize;

    assert_eq!(palette.len(), image_count);
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

            palette[idx].0 = pack_rgb(sum1.0 as u8, sum1.1 as u8, sum1.2 as u8);
            palette[idx].1 = pack_rgb(sum2.0 as u8, sum2.1 as u8, sum2.2 as u8);
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
    let mut idx = 0;
    src.chunks_exact(4).for_each(|block| {
        glyph.chunks_exact(4).for_each(|glyph| {
            let mut count = 0;
            for i in 0..4 {
                count += (block[i] ^ glyph[i]).count_zeros() as u32;
            }
            dst[idx] = count;
            idx += 1;
        })
    });
}

fn get_ascii_string(dst: &mut [u32], src: &[u32]) {
    let mut idx = 0;
    src.chunks_exact(ASCII_TABLE_SIZE as usize * 2)
        .for_each(|similarity| {
            let max = similarity
                .iter()
                .enumerate()
                .max_by_key(|(_idx, sim)| **sim)
                .unwrap()
                .0;
            dst[idx] = max as u32;
            idx += 1;
        });
}

struct VideoFrame {
    pub data: Vec<u32>,
    pub width: u32,
    pub height: u32,
}

fn open_stream(video_path: &str) -> tokio::sync::mpsc::Receiver<VideoFrame> {
    let (tx, rx) = tokio::sync::mpsc::channel(1);

    let video_path = video_path.to_owned();
    tokio::task::spawn_blocking(move || {
        ffmpeg_next::init().unwrap();
        let mut ictx = ffmpeg_next::format::input(&video_path).unwrap();
        let input = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or(ffmpeg_next::Error::StreamNotFound)
            .unwrap();
        let video_stream_index = input.index();

        let context_decoder =
            ffmpeg_next::codec::context::Context::from_parameters(input.parameters()).unwrap();
        let mut decoder = context_decoder.decoder().video().unwrap();

        use ffmpeg_next::software::scaling::Flags;
        let mut scaler = ffmpeg_next::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg_next::format::Pixel::RGBA,
            decoder.width(),
            decoder.height(),
            Flags::BICUBIC,
        )
        .unwrap();

        let mut frame = ffmpeg_next::frame::Video::empty();
        let mut rgb_frame = ffmpeg_next::frame::Video::empty();

        for (stream, packet) in ictx.packets() {
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet).unwrap();
                while decoder.receive_frame(&mut frame).is_ok() {
                    scaler.run(&frame, &mut rgb_frame).unwrap();
                    let rgb_data = rgb_frame
                        .data(0)
                        .chunks_exact(4)
                        .map(|chunk| pack_rgb(chunk[0], chunk[1], chunk[2]))
                        .collect::<Vec<_>>();

                    let frame_data = VideoFrame {
                        data: rgb_data,
                        width: rgb_frame.width(),
                        height: rgb_frame.height(),
                    };

                    if tx.blocking_send(frame_data).is_err() {
                        break;
                    }
                }
            }
        }
    });

    rx
}

fn get_cube_count(shape: [u32; 3], tiling: CubeDim) -> CubeCount {
    CubeCount::new_3d(
        shape[0].div_ceil(tiling.x),
        shape[1].div_ceil(tiling.y),
        shape[2].div_ceil(tiling.z),
    )
}

async fn render<RT: cubecl::prelude::Runtime>(
    client: &cubecl::client::ComputeClient<RT::Server, RT::Channel>,
    data: &[u32],
    width: u32,
    height: u32,
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
    bitmap_data: &[u32],
) -> anyhow::Result<(Vec<u32>, Vec<(u32, u32)>)> {
    let tiling = CubeDim::new_2d(16, 16);

    let (resized, resized_width, resized_height) = {
        // let _t = timer("resize");
        let resized_width = term_width * char_width;
        let resized_height = term_height * char_height;

        // let resized = alloc_tensor::<RT, u8>(client, vec![resized_height, resized_width, 3]);
        // let data = create_tensor::<RT, u8>(client, vec![height, width, 3], data);
        // let cube_count = get_cube_count([resized_height, resized_width, 1], tiling);

        // gpu::bicubic_resize::launch(
        //     client,
        //     cube_count,
        //     tiling,
        //     resized.as_arg(1),
        //     data.as_arg(1),
        //     resized_width,
        //     resized_height,
        //     width,
        //     height,
        // );
        // let resized = client.read_one_async(resized.handle.binding()).await;
        // (resized, resized_width, resized_height)

        let mut resized = vec![0; (resized_width * resized_height * 4) as usize];
        bicubic_resize(
            &mut resized,
            resized_width,
            resized_height,
            &data,
            width,
            height,
        );
        (resized, resized_width, resized_height)
    };

    let (repacked, repacked_width, repacked_height) = {
        // let _t = timer("repack");
        let repacked_width = resized_width;
        let repacked_height = resized_height;
        let mut repacked = vec![0; (repacked_width * repacked_height * 4) as usize];
        repack(
            &mut repacked,
            &resized,
            term_width,
            term_height,
            char_width,
            char_height,
        );
        (repacked, repacked_width, repacked_height)
    };

    let (clustered, palette, clustered_width, clustered_height) = {
        // let _t = timer("cluster");
        let clustered_width = resized_width;
        let clustered_height = resized_height;
        let mut clustered = vec![0; (term_width * term_height * 4) as usize];
        let mut palette = vec![(0, 0); (term_width * term_height) as usize];
        let mut brightness = vec![0; (resized_width * resized_height) as usize];
        compute_brightness(&mut brightness, &repacked);

        bimodal_luma_cluster(
            &mut clustered,
            &mut palette,
            &repacked,
            &brightness,
            term_width,
            term_height,
            char_width,
            char_height,
        );
        (clustered, palette, clustered_width, clustered_height)
    };

    let similarity = {
        // let _t = timer("similarity");
        let mut similarity = vec![0; (term_width * term_height * ASCII_TABLE_SIZE * 2) as usize];
        calc_similarity(
            &mut similarity,
            &clustered,
            &bitmap_data,
            term_width,
            term_height,
            char_width,
            char_height,
        );
        similarity
    };

    let term = {
        // let _t = timer("decide");
        let mut term = vec![0; (term_width * term_height) as usize];
        get_ascii_string(&mut term, &similarity);
        term
    };

    Ok((term, palette))
}

async fn screen(term: Vec<u32>, palette: Vec<(u32, u32)>, term_width: u32, term_height: u32) {
    let mut output = String::with_capacity((term_width * term_height * 20) as usize);
    let mut idx = 0;

    output.push_str("\x1b[0;0H"); // 移动到(0,0)

    for line in 0..term_height {
        for row in 0..term_width {
            let (fg_color, bg_color) = if term[idx] % 2 == 0 {
                (palette[idx].0, palette[idx].1)
            } else {
                (palette[idx].1, palette[idx].0)
            };
            let chr = char::from(((term[idx] / 2) + ASCII_START) as u8);

            let (fr, fg, fb) = unpack_rgb(fg_color);
            let (br, bg, bb) = unpack_rgb(bg_color);

            output.push_str(
                format!(
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m{}",
                    fr, fg, fb, br, bg, bb, chr
                )
                .as_str(),
            );

            idx += 1;
        }
        output.push_str("\x1b[0m\n"); // 重置颜色并换行
    }

    tokio::io::stdout()
        .write_all(output.as_bytes())
        .await
        .unwrap();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let file = std::env::args().skip(1).next().unwrap();
    let bitmap_data = generate_ascii_bitmap("font.otf").unwrap();
    println!("Generated {} bytes of bitmap data", bitmap_data.len());

    let (term_width, term_height) = crossterm::terminal::size()?;
    let term_width = term_width as u32 - 2;
    let term_height = term_height as u32 - 2;

    let char_width = 8;
    let char_height = 16;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<tokio::task::JoinHandle<_>>();

    tokio::task::spawn(async move {
        let frames = std::sync::atomic::AtomicU64::default();
        let start = std::time::Instant::now();
        while let Some(frame) = rx.recv().await {
            let (term, palette) = frame.await.unwrap();
            let now = std::time::Instant::now();
            let duration = (now - start).as_secs();
            let f = frames.load(std::sync::atomic::Ordering::Acquire);
            let magic = true;
            if rx.is_empty() || magic {
                screen(term, palette, term_width, term_height).await;
                tokio::io::stdout()
                    .write_all(
                        format!(
                            "ok, {duration} secs, {f} frames, {} fps",
                            f / (duration + 1)
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
            frames.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    });

    let mut stream = open_stream(&file);
    let bitmap_data = std::sync::Arc::new(bitmap_data);
    let client = std::sync::Arc::new(cubecl::wgpu::WgpuRuntime::client(&Default::default()));
    while let Some(frame) = stream.recv().await {
        let bitmap_data = std::sync::Arc::clone(&bitmap_data);
        let client = std::sync::Arc::clone(&client);
        let handle = tokio::task::spawn(async move {
            render::<cubecl::wgpu::WgpuRuntime>(
                &client,
                &frame.data,
                frame.width,
                frame.height,
                term_width,
                term_height,
                char_width,
                char_height,
                &bitmap_data,
            )
            .await
            .unwrap()
        });
        tx.send(handle).unwrap();
    }

    Ok(())
}
