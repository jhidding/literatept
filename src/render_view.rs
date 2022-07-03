#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use crate::{colour::RGBColour, Image};
use eframe::{
    egui,
    epaint::{Color32, ColorImage},
};
use egui::Rgba;
use egui_extras::RetainedImage;
use std::sync::mpsc::Receiver;

pub struct LitUi {
    // Receiver for (x, y, RGBColour)
    pub(crate) rx: Receiver<(usize, usize, RGBColour)>,
    pub(crate) img: ColorImage,
}

// `From` `RGBColour` to eguis `Rgba`
impl From<crate::colour::RGBColour> for Rgba {
    fn from(rgb: crate::colour::RGBColour) -> Self {
        let RGBColour(r, g, b) = rgb;
        Rgba::from_rgb(r as _, g as _, b as _)
    }
}

// Easily convert this crates `RGBColour` to egui `Color32`
impl From<crate::colour::RGBColour> for egui::Color32 {
    fn from(rgb: RGBColour) -> Self {
        let rgba: egui::Rgba = rgb.into();
        rgba.into()
    }
}

//  This crate has an Image object, but in egui's context it is more convenient to have
//  the data in egui-paint `ColorImage` format.
impl From<crate::Image> for ColorImage {
    fn from(img: crate::Image) -> Self {
        let Image {
            width: w,
            height: h,
            data: d,
        } = img;
        let pxs: Vec<Color32> = d.into_iter().map(|px| px.into()).collect();

        ColorImage {
            size: [w, h],
            pixels: pxs,
        }
    }
}

// This function is to be called from main.
pub fn render_window(lit_ui: LitUi) {
    let options = eframe::NativeOptions::default();
    eframe::run_native("Render view", options, Box::new(|_cc| Box::new(lit_ui)));
}

impl eframe::App for LitUi {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let img = &mut self.img;

        self.rx.try_iter().for_each(|(x, y, c)| {
            let idx = x + (y * img.width());
            let ci: Color32 = c.into();
            let cref = img.pixels.get_mut(idx).unwrap();
            *cref = ci;
            ctx.request_repaint();
        });

        let image = RetainedImage::from_color_image("render", img.clone()); // Place img in a Cell<T>

        egui::CentralPanel::default().show(ctx, |ui| {
            image.show_max_size(ui, ui.available_size());
        });
    }
}
