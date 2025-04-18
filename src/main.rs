use anyhow::Result;
use crossterm::{
    cursor::MoveToNextLine,
    execute,
    style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor},
};
use fontdue::{Font, FontSettings};
use std::fs;
use tokio::io::AsyncWriteExt;

#[inline]
fn timer(label: &str) -> scope_timer::ScopeTimer {
    scope_timer::ScopeTimer::new(label, scope_timer::TimeFormat::Milliseconds, None, false)
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

fn generate_ascii_bitmap(font_path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // 加载字体文件
    let font_data = fs::read(font_path)?;
    let font = Font::from_bytes(font_data, FontSettings::default())?;

    let mut result = Vec::new();
    // 处理可打印ASCII字符(32-126)
    for c in ASCII_START..=ASCII_END {
        let character = c as u8 as char;

        let bitmap = render_centered(&font, character);
        result.extend(bitmap.iter().flatten());
        result.extend(bitmap.iter().flatten().map(|x| 1 - x));
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

pub fn bicubic_resize(
    dst: &mut [u8],
    dst_width: u32,
    dst_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
) {
    // 验证输入参数有效性
    assert!(dst.len() >= (dst_width * dst_height * 3) as usize);
    assert!(src.len() >= (src_width * src_height * 3) as usize);

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

                    let idx = (py * src_width + px) as usize * 3;
                    let weight_x = cubic_weight(src_x - (x_floor + i) as f32, -0.5);
                    let weight_y = cubic_weight(src_y - (y_floor + j) as f32, -0.5);
                    let weight = weight_x * weight_y;

                    r += src[idx] as f32 * weight;
                    g += src[idx + 1] as f32 * weight;
                    b += src[idx + 2] as f32 * weight;
                    total_weight += weight;
                }
            }

            // 归一化并写入目标图像
            let dst_idx = (y * dst_width + x) as usize * 3;
            dst[dst_idx] = (r / total_weight).clamp(0.0, 255.0) as u8;
            dst[dst_idx + 1] = (g / total_weight).clamp(0.0, 255.0) as u8;
            dst[dst_idx + 2] = (b / total_weight).clamp(0.0, 255.0) as u8;
        }
    }
}

fn repack(
    dst: &mut [u8],
    src: &[u8],
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
                    let idx_in_start = (gh * (term_width * char_width) + gw) as usize * 3;
                    let idx_in_end = idx_in_start + 3;

                    let idx_out_start = (h * term_width * char_height * char_width
                        + w * char_height * char_width
                        + ch * char_width
                        + cw) as usize
                        * 3;
                    let idx_out_end = idx_out_start + 3;
                    (&mut dst[idx_out_start..idx_out_end])
                        .copy_from_slice(&src[idx_in_start..idx_in_end]);
                }
            }
        }
    }
}

fn _unpack(
    dst: &mut [u8],
    src: &[u8],
    palette: &[[[u8; 3]; 2]],
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
                    let idx_in_start = (gh * (term_width * char_width) + gw) as usize * 3;
                    let idx_in_end = idx_in_start + 3;

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

fn bimodal_luma_cluster(
    dst: &mut [u8],
    palette: &mut [[[u8; 3]; 2]],
    src: &[u8],
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
) {
    let px_per_img = (char_width * char_height) as usize;
    let image_count = (term_width * term_height) as usize;

    assert_eq!(palette.len(), image_count);
    assert_eq!(dst.len(), image_count * px_per_img);

    // 并行处理每张图像
    let mut idx = 0;
    std::iter::zip(
        src.chunks_exact(px_per_img * 3),
        dst.chunks_exact_mut(px_per_img),
    )
    .for_each(|(img, out)| {
        // 计算每个像素的亮度(考虑alpha)
        let luma: Vec<_> = img
            .chunks_exact(3)
            .map(|px| 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
            .collect();

        // 找到亮度中值
        let mut sorted = luma.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];

        // 计算两类颜色的平均值
        let (mut sum1, mut cnt1, mut sum2, mut cnt2) = ([0u32; 3], 0, [0u32; 3], 0);
        img.chunks_exact(3).zip(luma.iter()).for_each(|(px, &l)| {
            if l <= median {
                sum1.iter_mut().zip(px).for_each(|(s, &p)| *s += p as u32);
                cnt1 += 1;
            } else {
                sum2.iter_mut().zip(px).for_each(|(s, &p)| *s += p as u32);
                cnt2 += 1;
            }
        });

        match (cnt1, cnt2) {
            (0, 0) => unreachable!(),
            (0, cnt2) => {
                sum2.iter_mut().for_each(|x| *x /= cnt2);
                sum1 = sum2;
            }
            (cnt1, 0) => {
                sum1.iter_mut().for_each(|x| *x /= cnt1);
                sum2 = sum1;
            }
            (cnt1, cnt2) => {
                sum1.iter_mut().for_each(|x| *x /= cnt1);
                sum2.iter_mut().for_each(|x| *x /= cnt2);
            }
        };
        let sum1 = sum1.into_iter().map(|x| x as u8).collect::<Vec<_>>();
        let sum2 = sum2.into_iter().map(|x| x as u8).collect::<Vec<_>>();

        std::iter::zip(luma.iter(), out.iter_mut()).for_each(|(l, out)| {
            if *l <= median {
                *out = 0;
            } else {
                *out = 1;
            }
        });

        palette[idx][0].copy_from_slice(&sum1);
        palette[idx][1].copy_from_slice(&sum2);

        idx += 1;
    });
}

fn calc_similarity(
    dst: &mut [u8],
    src: &[u8],
    glyph: &[u8],
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
) {
    let mut idx = 0;
    let block_size = (char_width * char_height) as usize;
    src.chunks_exact(block_size).for_each(|block| {
        glyph.chunks_exact(block_size).for_each(|glyph| {
            dst[idx] = std::iter::zip(block, glyph)
                .filter(|(block, glyph)| block == glyph)
                .count() as u8;
            idx += 1;
        })
    });
}

fn get_ascii_string(dst: &mut [u8], src: &[u8]) {
    let mut idx = 0;
    src.chunks_exact(ASCII_TABLE_SIZE as usize * 2)
        .for_each(|similarity| {
            let max = similarity
                .iter()
                .enumerate()
                .max_by_key(|(_idx, sim)| **sim)
                .unwrap()
                .0;
            dst[idx] = max as u8;
            idx += 1;
        });
}

struct VideoFrame {
    pub data: Vec<u8>,
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
            ffmpeg_next::format::Pixel::RGB24,
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
                    let rgb_data = rgb_frame.data(0).to_vec();

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

async fn render(
    data: &[u8],
    width: u32,
    height: u32,
    term_width: u32,
    term_height: u32,
    char_width: u32,
    char_height: u32,
    bitmap_data: &[u8],
) -> anyhow::Result<(Vec<u8>, Vec<[[u8; 3]; 2]>)> {
    // assert_eq!(data.len() as u32, width * height * 3);
    // assert_eq!(width, term_width * char_width);
    // assert_eq!(height, term_height * char_height);
    // let (resized, resized_width, resized_height) = (
    //     data.to_vec(),
    //     term_width * char_width,
    //     term_height * char_height,
    // );

    let (resized, resized_width, resized_height) = {
        // let _t = timer("resize");
        let resized_width = term_width * char_width;
        let resized_height = term_height * char_height;
        let mut resized = vec![0; (resized_width * resized_height * 3) as usize];
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
        let mut repacked = vec![0; (repacked_width * repacked_height * 3) as usize];
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
        let mut clustered = vec![0; (clustered_width * clustered_height) as usize];
        let mut palette = vec![[[0; 3]; 2]; (term_width * term_height) as usize];
        bimodal_luma_cluster(
            &mut clustered,
            &mut palette,
            &repacked,
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

async fn screen(term: Vec<u8>, palette: Vec<[[u8; 3]; 2]>, term_width: u32, term_height: u32) {
    let mut output = String::with_capacity((term_width * term_height * 20) as usize);
    let mut idx = 0;

    output.push_str("\x1b[0;0H"); // 移动到(0,0)

    for line in 0..term_height {
        for row in 0..term_width {
            let fg_color = palette[(line * term_width + row) as usize][(term[idx] % 2) as usize];
            let bg_color =
                palette[(line * term_width + row) as usize][(1 - term[idx] % 2) as usize];
            let chr = char::from((term[idx] / 2) + ASCII_START as u8);

            output.push_str(
                format!(
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m{}",
                    fg_color[0],
                    fg_color[1],
                    fg_color[2],
                    bg_color[0],
                    bg_color[1],
                    bg_color[2],
                    chr
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
            if rx.is_empty() {
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
    while let Some(frame) = stream.recv().await {
        let bitmap_data = std::sync::Arc::clone(&bitmap_data);
        let handle = tokio::task::spawn(async move {
            render(
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
