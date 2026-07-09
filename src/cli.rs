use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use rsomics_common::{CommonFlags, Result, RsomicsError, Tool, ToolMeta};
use rsomics_help::{Example, FlagSpec, HelpSpec, Origin, Section};

use rsomics_sc_embedding_density::{Config, read_embedding, read_groups, run};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Parser, Debug)]
#[command(name = "rsomics-sc-embedding-density", version, about, long_about = None, disable_help_flag = true)]
pub struct Cli {
    /// N×2 embedding (whitespace/TSV; a non-numeric first line is a header).
    #[arg(long)]
    embedding: PathBuf,

    /// Optional N-length per-cell group labels, one per line, in cell order.
    #[arg(long)]
    groups: Option<PathBuf>,

    /// Decimal places in the output.
    #[arg(long, default_value_t = 12)]
    precision: usize,

    /// Output path; writes stdout when "-".
    #[arg(short = 'o', long, default_value = "-")]
    output: String,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Tool for Cli {
    fn meta() -> ToolMeta {
        META
    }
    fn common(&self) -> &CommonFlags {
        &self.common
    }

    fn execute(self) -> Result<()> {
        self.common.install_rayon_pool()?;

        let emb = read_embedding(BufReader::new(File::open(&self.embedding).map_err(
            |e| RsomicsError::InvalidInput(format!("{}: {e}", self.embedding.display())),
        )?))?;

        let groups = match &self.groups {
            Some(p) => {
                let r =
                    BufReader::new(File::open(p).map_err(|e| {
                        RsomicsError::InvalidInput(format!("{}: {e}", p.display()))
                    })?);
                Some(read_groups(r, emb.x.len())?)
            }
            None => None,
        };

        let mut out: Box<dyn Write> = if self.output == "-" && self.common.json {
            Box::new(BufWriter::new(std::io::sink()))
        } else if self.output == "-" {
            Box::new(BufWriter::new(std::io::stdout().lock()))
        } else {
            Box::new(BufWriter::new(
                File::create(&self.output).map_err(RsomicsError::Io)?,
            ))
        };

        run(
            &emb,
            groups.as_deref(),
            &mut out,
            &Config {
                precision: self.precision,
            },
        )?;
        out.flush().map_err(RsomicsError::Io)
    }
}

pub static HELP: HelpSpec = HelpSpec {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
    tagline: "Per-group Gaussian-KDE cell density on a 2D embedding, scaled to [0,1].",
    origin: Some(Origin {
        upstream: "scanpy sc.tl.embedding_density",
        upstream_license: "BSD-3-Clause",
        our_license: "MIT OR Apache-2.0",
        paper_doi: Some("10.1186/s13059-017-1382-0"),
    }),
    usage_lines: &[
        "--embedding emb.tsv [-o density.tsv]",
        "--embedding emb.tsv --groups grp.tsv",
    ],
    sections: &[Section {
        title: "OPTIONS",
        flags: &[
            FlagSpec {
                short: None,
                long: "embedding",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("Path"),
                required: true,
                default: None,
                description: "N×2 embedding (whitespace/TSV; non-numeric first line is a header).",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "groups",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("Path"),
                required: false,
                default: None,
                description: "Per-cell group labels, one per line, in cell order.",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "precision",
                aliases: &[],
                value: Some("<int>"),
                type_hint: Some("usize"),
                required: false,
                default: Some("12"),
                description: "Decimal places in the output.",
                why_default: None,
            },
            FlagSpec {
                short: Some('o'),
                long: "output",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: Some("-"),
                description: "Output path (- for stdout).",
                why_default: None,
            },
        ],
    }],
    examples: &[
        Example {
            description: "Whole-embedding density",
            command: "rsomics-sc-embedding-density --embedding umap.tsv",
        },
        Example {
            description: "Per-group density, 8 threads, to a file",
            command: "rsomics-sc-embedding-density --embedding umap.tsv --groups phase.tsv -t 8 -o d.tsv",
        },
    ],
    json_result_schema_doc: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        Cli::command().debug_assert();
    }
}
