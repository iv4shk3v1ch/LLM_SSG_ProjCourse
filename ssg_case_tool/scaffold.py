from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, slug_to_path, write_text

SUPPORTED_SSGS = {"eleventy", "hugo", "zola", "pelican"}


def _normalized_pages(spec: dict[str, Any]) -> list[dict[str, str]]:
    pages = spec.get("pages", [])
    normalized: list[dict[str, str]] = []
    for page in pages:
        title = str(page.get("title", "Untitled"))
        slug = str(page.get("slug", "")).strip("/")
        placeholder = str(page.get("placeholder", f"{title} text goes here."))
        normalized.append({"title": title, "slug": slug, "placeholder": placeholder})
    if not normalized:
        normalized = [{"title": "Home", "slug": "", "placeholder": "Home text goes here."}]
    return normalized


def _normalized_nav(spec: dict[str, Any], pages: list[dict[str, str]]) -> list[dict[str, str]]:
    nav = spec.get("nav")
    if isinstance(nav, list) and nav:
        return [{"label": str(it["label"]), "url": str(it["url"])} for it in nav]
    fallback = []
    for page in pages:
        url = "/" if not page["slug"] else f"/{page['slug']}/"
        fallback.append({"label": page["title"], "url": url})
    return fallback


def scaffold_project(
    spec: dict[str, Any],
    out_dir: Path,
    ssg: str,
    force: bool = False,
    spec_base_dir: Path | None = None,
) -> dict[str, Any]:
    ssg_name = ssg.lower()
    if ssg_name not in SUPPORTED_SSGS:
        raise ValueError(f"Unsupported SSG: {ssg_name}. Supported: {sorted(SUPPORTED_SSGS)}")

    site = spec.get("site", {})
    site_name = str(site.get("name", "Static Site Prototype"))
    base_url = str(site.get("base_url", "https://example.com"))
    theme = str(site.get("theme", "baseline"))
    deployment_target = str(site.get("deployment_target", "static-host"))

    pages = _normalized_pages(spec)
    nav = _normalized_nav(spec, pages)
    ensure_dir(out_dir)

    files = _files_for_ssg(
        ssg=ssg_name,
        site_name=site_name,
        base_url=base_url,
        theme=theme,
        deployment_target=deployment_target,
        pages=pages,
        nav=nav,
    )

    written: list[str] = []
    for rel_path, content in files.items():
        full = out_dir / rel_path
        write_text(full, content, force=force)
        written.append(str(rel_path))

    migration = spec.get("migration", {})
    migration_artifacts = _generate_migration_artifacts(
        out_dir=out_dir,
        migration=migration,
        ssg=ssg_name,
        pages=pages,
        force=force,
        spec_base_dir=spec_base_dir or out_dir,
    )
    written.extend(migration_artifacts)

    return {
        "ssg": ssg_name,
        "site_name": site_name,
        "base_url": base_url,
        "theme": theme,
        "deployment_target": deployment_target,
        "files_written": sorted(written),
    }


def _files_for_ssg(
    ssg: str,
    site_name: str,
    base_url: str,
    theme: str,
    deployment_target: str,
    pages: list[dict[str, str]],
    nav: list[dict[str, str]],
) -> dict[Path, str]:
    if ssg == "eleventy":
        return _eleventy_files(site_name, base_url, theme, deployment_target, pages, nav)
    if ssg == "hugo":
        return _hugo_files(site_name, base_url, theme, deployment_target, pages, nav)
    if ssg == "zola":
        return _zola_files(site_name, base_url, theme, deployment_target, pages, nav)
    return _pelican_files(site_name, base_url, theme, deployment_target, pages, nav)


def _nav_html(nav: list[dict[str, str]]) -> str:
    lines = ['<nav aria-label="Main navigation">', "  <ul>"]
    for item in nav:
        lines.append(f'    <li><a href="{item["url"]}">{item["label"]}</a></li>')
    lines.extend(["  </ul>", "</nav>"])
    return "\n".join(lines)


def _eleventy_page(page: dict[str, str]) -> tuple[Path, str]:
    slug = page["slug"]
    file_name = "index.md" if not slug else f"{slug_to_path(slug)}.md"
    permalink = "/" if not slug else f"/{slug}/"
    body = (
        "---\n"
        "layout: layouts/base.njk\n"
        f'title: "{page["title"]}"\n'
        f'permalink: "{permalink}"\n'
        "---\n\n"
        f'{page["placeholder"]}\n'
    )
    return Path("src") / file_name, body


def _eleventy_files(
    site_name: str,
    base_url: str,
    theme: str,
    deployment_target: str,
    pages: list[dict[str, str]],
    nav: list[dict[str, str]],
) -> dict[Path, str]:
    files: dict[Path, str] = {
        Path("package.json"): (
            "{\n"
            '  "name": "ssg-prototype-eleventy",\n'
            '  "private": true,\n'
            '  "scripts": {\n'
            '    "build": "npx @11ty/eleventy",\n'
            '    "serve": "npx @11ty/eleventy --serve"\n'
            "  }\n"
            "}\n"
        ),
        Path(".eleventy.js"): (
            "module.exports = function(eleventyConfig) {\n"
            "  eleventyConfig.addPassthroughCopy({ \"public\": \"/\" });\n"
            "  return {\n"
            "    dir: {\n"
            "      input: \"src\",\n"
            "      includes: \"_includes\",\n"
            "      output: \"_site\"\n"
            "    }\n"
            "  };\n"
            "};\n"
        ),
        Path("src/_data/site.json"): (
            "{\n"
            f'  "name": "{site_name}",\n'
            f'  "baseUrl": "{base_url}",\n'
            f'  "theme": "{theme}",\n'
            f'  "deploymentTarget": "{deployment_target}"\n'
            "}\n"
        ),
        Path("src/_includes/layouts/base.njk"): (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
            "  <title>{{ title }} | {{ site.name }}</title>\n"
            "</head>\n"
            "<body>\n"
            "  {% include \"partials/nav.njk\" %}\n"
            "  <main>\n"
            "    <h1>{{ title }}</h1>\n"
            "    {{ content | safe }}\n"
            "  </main>\n"
            "</body>\n"
            "</html>\n"
        ),
        Path("src/_includes/partials/nav.njk"): _nav_html(nav) + "\n",
        Path("frontmatter_templates/eleventy_frontmatter.md"): (
            "---\n"
            "layout: layouts/base.njk\n"
            "title: \"Page Title\"\n"
            "permalink: \"/page-slug/\"\n"
            "---\n\n"
            "Page text goes here.\n"
        ),
    }
    for page in pages:
        rel, body = _eleventy_page(page)
        files[rel] = body
    return files


def _hugo_page(page: dict[str, str]) -> tuple[Path, str]:
    slug = page["slug"]
    file_name = "_index.md" if not slug else f"{slug_to_path(slug)}.md"
    body = (
        "+++\n"
        f'title = "{page["title"]}"\n'
        "+++\n\n"
        f'{page["placeholder"]}\n'
    )
    return Path("content") / file_name, body


def _hugo_files(
    site_name: str,
    base_url: str,
    theme: str,
    deployment_target: str,
    pages: list[dict[str, str]],
    nav: list[dict[str, str]],
) -> dict[Path, str]:
    files: dict[Path, str] = {
        Path("config.toml"): (
            f'baseURL = "{base_url}"\n'
            'languageCode = "en-us"\n'
            f'title = "{site_name}"\n'
            "[params]\n"
            f'  theme = "{theme}"\n'
            f'  deployment_target = "{deployment_target}"\n'
        ),
        Path("layouts/_default/baseof.html"): (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
            "  <title>{{ .Title }} | {{ .Site.Title }}</title>\n"
            "</head>\n"
            "<body>\n"
            "  {{ partial \"nav.html\" . }}\n"
            "  <main>{{ block \"main\" . }}{{ end }}</main>\n"
            "</body>\n"
            "</html>\n"
        ),
        Path("layouts/_default/single.html"): (
            "{{ define \"main\" }}\n"
            "  <h1>{{ .Title }}</h1>\n"
            "  {{ .Content }}\n"
            "{{ end }}\n"
        ),
        Path("layouts/index.html"): (
            "{{ define \"main\" }}\n"
            "  <h1>{{ .Site.Title }}</h1>\n"
            "  {{ .Content }}\n"
            "{{ end }}\n"
        ),
        Path("layouts/partials/nav.html"): _nav_html(nav) + "\n",
        Path("frontmatter_templates/hugo_frontmatter.md"): (
            "+++\n"
            'title = "Page Title"\n'
            "+++\n\n"
            "Page text goes here.\n"
        ),
    }
    for page in pages:
        rel, body = _hugo_page(page)
        files[rel] = body
    return files


def _zola_page(page: dict[str, str]) -> tuple[Path, str]:
    slug = page["slug"]
    file_name = "_index.md" if not slug else f"{slug_to_path(slug)}.md"
    body = (
        "+++\n"
        f'title = "{page["title"]}"\n'
        "template = \"page.html\"\n"
        "+++\n\n"
        f'{page["placeholder"]}\n'
    )
    return Path("content") / file_name, body


def _zola_files(
    site_name: str,
    base_url: str,
    theme: str,
    deployment_target: str,
    pages: list[dict[str, str]],
    nav: list[dict[str, str]],
) -> dict[Path, str]:
    files: dict[Path, str] = {
        Path("config.toml"): (
            f'base_url = "{base_url}"\n'
            f'title = "{site_name}"\n'
            "build_search_index = false\n\n"
            "[extra]\n"
            f'theme = "{theme}"\n'
            f'deployment_target = "{deployment_target}"\n'
        ),
        Path("templates/base.html"): (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
            "  <title>{% block title %}{{ config.title }}{% endblock title %}</title>\n"
            "</head>\n"
            "<body>\n"
            "  {% include \"partials/nav.html\" %}\n"
            "  <main>{% block content %}{% endblock content %}</main>\n"
            "</body>\n"
            "</html>\n"
        ),
        Path("templates/page.html"): (
            "{% extends \"base.html\" %}\n"
            "{% block title %}{{ page.title }} | {{ config.title }}{% endblock title %}\n"
            "{% block content %}\n"
            "<h1>{{ page.title }}</h1>\n"
            "{{ page.content | safe }}\n"
            "{% endblock content %}\n"
        ),
        Path("templates/index.html"): (
            "{% extends \"base.html\" %}\n"
            "{% block content %}\n"
            "<h1>{{ section.title }}</h1>\n"
            "{{ section.content | safe }}\n"
            "{% endblock content %}\n"
        ),
        Path("templates/partials/nav.html"): _nav_html(nav) + "\n",
        Path("frontmatter_templates/zola_frontmatter.md"): (
            "+++\n"
            'title = "Page Title"\n'
            "template = \"page.html\"\n"
            "+++\n\n"
            "Page text goes here.\n"
        ),
    }
    for page in pages:
        rel, body = _zola_page(page)
        files[rel] = body
    return files


def _pelican_page(page: dict[str, str]) -> tuple[Path, str]:
    slug = page["slug"]
    file_name = "index.md" if not slug else f"{slug_to_path(slug)}.md"
    body = (
        f"Title: {page['title']}\n"
        f"Slug: {slug_to_path(slug)}\n"
        "Status: draft\n\n"
        f'{page["placeholder"]}\n'
    )
    return Path("content/pages") / file_name, body


def _pelican_files(
    site_name: str,
    base_url: str,
    theme: str,
    deployment_target: str,
    pages: list[dict[str, str]],
    nav: list[dict[str, str]],
) -> dict[Path, str]:
    files: dict[Path, str] = {
        Path("pelicanconf.py"): (
            f"SITENAME = '{site_name}'\n"
            f"SITEURL = '{base_url}'\n"
            "PATH = 'content'\n"
            "TIMEZONE = 'UTC'\n"
            "DEFAULT_LANG = 'en'\n"
            "THEME = 'themes/custom'\n"
            f"CUSTOM_THEME_NAME = '{theme}'\n"
            f"DEPLOYMENT_TARGET = '{deployment_target}'\n"
        ),
        Path("themes/custom/templates/base.html"): (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
            "  <title>{% block title %}{{ SITENAME }}{% endblock %}</title>\n"
            "</head>\n"
            "<body>\n"
            "  {% include 'partials/nav.html' %}\n"
            "  <main>{% block content %}{% endblock %}</main>\n"
            "</body>\n"
            "</html>\n"
        ),
        Path("themes/custom/templates/partials/nav.html"): _nav_html(nav) + "\n",
        Path("frontmatter_templates/pelican_frontmatter.md"): (
            "Title: Page Title\n"
            "Slug: page-slug\n"
            "Status: draft\n\n"
            "Page text goes here.\n"
        ),
    }
    for page in pages:
        rel, body = _pelican_page(page)
        files[rel] = body
    return files


def _migration_wrapper_header(ssg: str, title: str, slug: str) -> str:
    if ssg in {"hugo", "zola"}:
        return f'+++\ntitle = "{title}"\nslug = "{slug}"\n+++\n\n'
    if ssg == "pelican":
        return f"Title: {title}\nSlug: {slug}\nStatus: draft\n\n"
    return f'---\ntitle: "{title}"\npermalink: "/{slug}/"\n---\n\n'


def _generate_migration_artifacts(
    out_dir: Path,
    migration: Any,
    ssg: str,
    pages: list[dict[str, str]],
    force: bool,
    spec_base_dir: Path,
) -> list[str]:
    if not isinstance(migration, dict):
        return []

    source_pages = migration.get("source_pages")
    if not isinstance(source_pages, list) or not source_pages:
        return []

    written: list[str] = []
    table_lines = [
        "# Migration Mapping Plan",
        "",
        "| Source Path | Target Slug | Template Strategy |",
        "|---|---|---|",
    ]

    wrap_dir = out_dir / "migration_wrapped_content"
    ensure_dir(wrap_dir)

    for item in source_pages:
        source_path = str(item.get("source_path", "/unknown"))
        title = str(item.get("title", "Untitled"))
        target_slug = str(item.get("target_slug", slug_to_path(title.lower().replace(" ", "-"))))
        text_file = item.get("text_file")
        table_lines.append(f"| {source_path} | /{target_slug}/ | frontmatter wrapper + provided text |")

        body = f"{title} text goes here.\n"
        if text_file:
            text_path = spec_base_dir / str(text_file)
            if text_path.exists():
                body = text_path.read_text(encoding="utf-8")

        wrapped = _migration_wrapper_header(ssg=ssg, title=title, slug=target_slug) + body.strip() + "\n"
        out_file = wrap_dir / f"{slug_to_path(target_slug)}.md"
        write_text(out_file, wrapped, force=force)
        written.append(str(out_file.relative_to(out_dir)))

    mapping_md = out_dir / "migration_mapping.md"
    write_text(mapping_md, "\n".join(table_lines) + "\n", force=force)
    written.append(str(mapping_md.relative_to(out_dir)))

    # Minimal traceability file links migration wrappers to generated pages.
    linkage = out_dir / "migration_linkage.txt"
    page_slugs = [p["slug"] or "index" for p in pages]
    linkage_text = "Generated page slugs:\n" + "\n".join(f"- {slug}" for slug in page_slugs) + "\n"
    write_text(linkage, linkage_text, force=force)
    written.append(str(linkage.relative_to(out_dir)))

    return written

