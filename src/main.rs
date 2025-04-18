use cubecl::Runtime;
use cubecl::prelude::*;
use fontdue::{Font, FontSettings};
use tokio::io::AsyncWriteExt;

const ASCII_START: u32 = 32;
const ASCII_END: u32 = 126;
const ASCII_TABLE_SIZE: u32 = ASCII_END - ASCII_START + 1;

#[no_implicit_prelude]
mod gpu {
    extern crate cubecl;
    extern crate std;

    use cubecl::prelude::*;
    use std::clone::Clone;
    use std::convert::Into;
    use std::default::Default;

    use crate::ASCII_TABLE_SIZE;

    #[cube]
    fn pack_rgb(r: u32, g: u32, b: u32) -> u32 {
        b << 16 | g << 8 | r
    }

    #[cube]
    fn unpack_rgb(packed: u32) -> (u32, u32, u32) {
        let b = packed >> 16;
        let g = (packed >> 8) & 0xff;
        let r = packed & 0xff;
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

        let x_ratio = src_width as f32 / dst_width as f32;
        let y_ratio = src_height as f32 / dst_height as f32;

        let src_x = x as f32 * x_ratio;
        let src_y = y as f32 * y_ratio;

        let x_floor = f32::floor(src_x) as i32;
        let y_floor = f32::floor(src_y) as i32;

        // 计算16个相邻像素的权重
        let mut r = 0.0;
        let mut g = 0.0;
        let mut b = 0.0;
        let mut total_weight = 0.0;

        for i in -1..3 {
            for j in -1..3 {
                let px = i32::clamp(x_floor + i, 0, src_width as i32 - 1) as u32;
                let py = i32::clamp(y_floor + j, 0, src_height as i32 - 1) as u32;

                let idx = py * src_width + px;
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

        (for i in 0..4 {
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
        });

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
        let brightness = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
        dst[idx] = brightness as u32;
    }

    #[cube]
    fn count_zeros(x: u32) -> u32 {
        let x = x - ((x >> 1) & 0x55555555);
        let x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        let x = (x + (x >> 4)) & 0x0F0F0F0F;
        let x = x + (x >> 8);
        let x = x + (x >> 16);
        let ones = x & 0x0000003F;

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
    b << 16 | g << 8 | r
}

fn unpack_rgb(packed: u32) -> (u8, u8, u8) {
    let b = packed >> 16;
    let g = (packed >> 8) & 0xff;
    let r = packed & 0xff;
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
    glyph: &cubecl::server::Handle,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let tiling = CubeDim::new_2d(16, 16);

    let (resized, resized_width, resized_height) = {
        // let _t = timer("resize");
        let resized_width = term_width * char_width;
        let resized_height = term_height * char_height;
        let resized_len = (resized_height * resized_width) as usize;

        let resized = client.empty(size_of::<u32>() * resized_len);
        let input = client.create(u32::as_bytes(data));

        unsafe {
            gpu::bicubic_resize::launch::<RT>(
                client,
                get_cube_count([resized_width, resized_height, 1], tiling),
                tiling,
                ArrayArg::from_raw_parts::<u32>(&resized, resized_len, 1),
                ArrayArg::from_raw_parts::<u32>(&input, data.len(), 1),
                ScalarArg::new(resized_width),
                ScalarArg::new(resized_height),
                ScalarArg::new(width),
                ScalarArg::new(height),
            );
        };
        (resized, resized_width, resized_height)
    };

    let (repacked, repacked_width, repacked_height) = {
        // let _t = timer("repack");

        let repacked_width = resized_width;
        let repacked_height = resized_height;
        let repacked = client.empty(size_of::<u32>() * (repacked_width * repacked_height) as usize);
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
            client.empty(size_of::<u32>() * (repacked_height * repacked_width) as usize);
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
        let clustered = client.empty(size_of::<u32>() * (term_width * term_height * 4) as usize);
        let palette = client.empty(size_of::<u32>() * (term_width * term_height * 2) as usize);
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
            .empty(size_of::<u32>() * (term_width * term_height * ASCII_TABLE_SIZE * 2) as usize);
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
        let term = client.empty(size_of::<u32>() * (term_width * term_height) as usize);
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

async fn screen(term: Vec<u32>, palette: Vec<u32>, term_width: u32, term_height: u32) {
    let mut output = String::with_capacity((term_width * term_height * 20) as usize);
    let mut idx = 0;

    output.push_str("\x1b[0;0H"); // 移动到(0,0)

    for _line in 0..term_height {
        for _row in 0..term_width {
            let (fg_color, bg_color) = if term[idx] % 2 == 0 {
                (palette[idx * 2], palette[idx * 2 + 1])
            } else {
                (palette[idx * 2 + 1], palette[idx * 2])
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
    let file = std::env::args().nth(1).expect("usage: $0 <video_path>");
    let bitmap_data = generate_ascii_bitmap("font.otf").unwrap();
    println!("Generated {} bytes of bitmap data", bitmap_data.len());

    let (term_width, term_height) = crossterm::terminal::size()?;
    let term_width = term_width as u32 - 2;
    let term_height = term_height as u32 - 2;

    let char_width = 8;
    let char_height = 16;

    let (tx, mut rx) = tokio::sync::mpsc::channel::<tokio::task::JoinHandle<_>>(4);

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
    let client = std::sync::Arc::new(cubecl::wgpu::WgpuRuntime::client(&Default::default()));
    let glyph = std::sync::Arc::new(client.create(u32::as_bytes(&bitmap_data)));
    while let Some(frame) = stream.recv().await {
        let glyph = std::sync::Arc::clone(&glyph);
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
                &glyph,
            )
            .await
            .unwrap()
        });
        tx.send(handle).await.unwrap();
    }

    Ok(())
}
