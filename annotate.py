# MdBook preprocessor for annotating code blocks
import json
import sys
import mawk
import re


class Annotator(mawk.RuleSet):
    @mawk.on_match(r"```([^ ]+) +#([^ ]+)")
    def annotate_named_ref(self, m: re.Match):
        return [f"<div class=\"codemark\">«{m.group(2)}»</div>","",
                f"```{m.group(1)}"]

    @mawk.on_match(r"```([^ ]+) +file=([^ ]+)")
    def annotate_file_ref(self, m: re.Match):
        return [f"<div class=\"codemark\">file:{m.group(2)}</div>","",
                f"```{m.group(1)}"]


def content_pass(f, content):
    if isinstance(content, dict) and "content" in content:
        content["content"] = f(content["content"])
    elif isinstance(content, list):
        for item in content:
            content_pass(f, item)
    elif isinstance(content, dict):
        for item in content.values():
            content_pass(f, item)
    else:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "supports":
        sys.exit(0)

    context, book = json.load(sys.stdin)
    content_pass(Annotator().run, book)
    print(json.dumps(book))
