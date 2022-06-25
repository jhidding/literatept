# This is a suggestion for a Makefile. This assumes you have executed,
#
#     git submodule add git@github.com:entangled/bootstrap-submodule bootstrap
#
# and that you have your literate sources in `./lit`.
#
# Make sure you have the following things installed:
#
#   - Entangled (the daemon)
#   - entangled-filters (the pandoc filters: pip install ...)
#   - Pandoc
#   - BrowserSync (npm install -g ...)
#   - InotifyTools (available from most GNU/Linux distributions)
#
# The website will be built in `./docs`, from which it can be served as
# github.io pages.
#
#
# You should list the Markdown sources here in the order that they should
# appear.
# input_files := lit/index.md
theme := default
theme_dir := .entangled/templates/$(theme)

pandoc_args += -s -t html5 -f markdown+fenced_code_attributes --toc --toc-depth 2
pandoc_args += --template $(theme_dir)/template.html
pandoc_args += --css theme.css
pandoc_args += --mathjax
# pandoc_args += --syntax-definition .entangled/syntax/dhall.xml
# pandoc_args += --highlight-style $(theme_dir)/syntax.theme
pandoc_args += --section-divs
pandoc_args += --lua-filter .entangled/scripts/hide.lua
pandoc_args += --lua-filter .entangled/scripts/annotate.lua
pandoc_args += --lua-filter .entangled/scripts/make.lua
pandoc_input := $(wildcard lit/*.md)
pandoc_output := docs/index.html

static_files := $(theme_dir)/theme.css $(theme_dir)/static
static_targets := $(static_files:$(theme_dir)/%=docs/%)
functional_deps := Makefile $(wildcard .entangled/scripts/*.lua) $(theme_dir)/template.html $(theme_dir)/syntax.theme
image_srcs := $(wildcard lit/img/*)
images := $(image_srcs:lit/img/%=docs/img/%)

site: $(pandoc_output) $(static_targets) $(figure_targets) $(images)

clean:
	rm -rf docs

$(images): docs/img/%: lit/img/%
	@mkdir -p $(@D)
	cp -r $< $@

$(static_targets): docs/%: $(theme_dir)/%
	@mkdir -p $(@D)
	rm -rf $@
	cp -r $< $@

docs/index.html: $(pandoc_input) $(functional_deps)
	@mkdir -p $(@D)
	pandoc $(pandoc_args) -o $@ $(pandoc_input)

# Starts a tmux with Entangled, Browser-sync and an Inotify loop for running
# Pandoc.
watch:
	@tmux new-session make --no-print-directory watch-pandoc \; \
		split-window -v make --no-print-directory watch-browser-sync \; \
		split-window -v entangled daemon \; \
		select-layout even-vertical \;

watch-pandoc:
	@while true; do \
		inotifywait -e close_write -r .entangled Makefile README.md ; \
		make site; \
	done

watch-browser-sync:
	browser-sync start -w -s docs

