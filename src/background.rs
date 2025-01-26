use clap::{
    builder::{TypedValueParser, ValueParserFactory},
    error::ErrorKind,
};
use itertools::Itertools;
use pix::{el::Pixel, rgb::Rgba8p};
use std::{fmt::Display, str::FromStr};

#[derive(Debug, Clone)]
pub struct Background(pub Rgba8p);

pub struct BackgroundError();

impl Display for BackgroundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error parsing color")
    }
}

impl Display for Background {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r: u8 = self.0.one().into();
        let g: u8 = self.0.two().into();
        let b: u8 = self.0.three().into();

        write!(f, "{:02x}{:02x}{:02x}", r, g, b)
    }
}

impl FromStr for Background {
    type Err = BackgroundError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 6 {
            return Err(BackgroundError());
        }

        s.chars()
            .chunks(2)
            .into_iter()
            .map(Iterator::collect::<String>)
            .map(|c| u8::from_str_radix(&c, 16))
            .collect::<Result<Vec<u8>, _>>()
            .map_err(|_| BackgroundError())
            .map(|rgb| Self(Rgba8p::new(rgb[0], rgb[1], rgb[2], 255)))
    }
}

#[derive(Debug, Clone)]
pub struct BackgroundParser;

impl TypedValueParser for BackgroundParser {
    type Value = Background;

    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        _arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let value_str = value
            .to_str()
            .ok_or_else(|| clap::Error::raw(ErrorKind::InvalidUtf8, "Invalid UTF-8"))?;

        Background::from_str(value_str).map_err(|e| clap::Error::raw(ErrorKind::ValueValidation, e))
    }
}

impl ValueParserFactory for Background {
    type Parser = BackgroundParser;

    fn value_parser() -> Self::Parser {
        BackgroundParser
    }
}
