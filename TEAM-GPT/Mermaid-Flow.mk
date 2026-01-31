# Mermaid-Flow.mk
# Build diagrams from Mermaid source files for TEAM-GPT docs
#
# Usage:
#   make all       # build all diagrams (svg/png/pdf)
#   make svg       # build only svg
#   make png       # build only png
#   make pdf       # build only pdf
#   make clean     # remove old diagrams

# Require mermaid-cli installed:
#   npm install -g @mermaid-js/mermaid-cli

MMDC   := mmdc
SRCEXT := mmd mkd md
SRCDIR := diagrams
OUTDIR := diagrams/out

# All mermaid source files in directories
MMD_SRCS := $(wildcard $(SRCDIR)/*.mmd) $(wildcard $(SRCDIR)/*.md)

# Targets
SVG_OUT := $(patsubst $(SRCDIR)/%.mmd,$(OUTDIR)/%.svg,$(wildcard $(SRCDIR)/*.mmd))
PNG_OUT := $(patsubst $(SRCDIR)/%.mmd,$(OUTDIR)/%.png,$(wildcard $(SRCDIR)/*.mmd))
PDF_OUT := $(patsubst $(SRCDIR)/%.mmd,$(OUTDIR)/%.pdf,$(wildcard $(SRCDIR)/*.mmd))

all: svg png pdf

# SVG build
svg: $(SVG_OUT)

$(OUTDIR)/%.svg: $(SRCDIR)/%.mmd
	@mkdir -p $(OUTDIR)
	@echo "🔹 Generating SVG diagram $@"
	$(MMDC) --input $< --output $@ --quiet

# PNG build
png: $(PNG_OUT)

$(OUTDIR)/%.png: $(SRCDIR)/%.mmd
	@mkdir -p $(OUTDIR)
	@echo "🔸 Generating PNG diagram $@"
	$(MMDC) --input $< --output $@ --png

# PDF build
pdf: $(PDF_OUT)

$(OUTDIR)/%.pdf: $(SRCDIR)/%.mmd
	@mkdir -p $(OUTDIR)
	@echo "📄 Generating PDF diagram $@"
	$(MMDC) --input $< --output $@ --pdfFit

clean:
	@echo "🧹 Cleaning output diagrams"
	@rm -rf $(OUTDIR)/*

.PHONY: all svg png pdf clean
