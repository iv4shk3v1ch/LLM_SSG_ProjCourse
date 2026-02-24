from __future__ import annotations

from typing import Any


def validate_spec_schema(spec: dict[str, Any], supported_ssgs: set[str]) -> dict[str, Any]:
    errors: list[dict[str, str]] = []

    def add_error(code: str, path: str, message: str) -> None:
        errors.append({"code": code, "path": path, "message": message})

    if not isinstance(spec, dict):
        add_error("invalid_type", "$", "Spec root must be an object.")
        return _validation_result(errors)

    site = spec.get("site")
    if not isinstance(site, dict):
        add_error("missing_site", "$.site", "Field `site` must be an object.")
    else:
        for field in ("name", "type", "theme", "deployment_target", "base_url"):
            value = site.get(field)
            if not isinstance(value, str) or not value.strip():
                add_error("invalid_site_field", f"$.site.{field}", "Must be a non-empty string.")

    ssg = spec.get("ssg")
    preferred = None
    candidates: list[str] = []
    if not isinstance(ssg, dict):
        add_error("missing_ssg", "$.ssg", "Field `ssg` must be an object.")
    else:
        preferred_raw = ssg.get("preferred")
        if not isinstance(preferred_raw, str) or not preferred_raw.strip():
            add_error("invalid_ssg_preferred", "$.ssg.preferred", "Must be a non-empty string.")
        else:
            preferred = preferred_raw.strip().lower()
            if preferred not in supported_ssgs:
                add_error(
                    "unsupported_ssg_preferred",
                    "$.ssg.preferred",
                    f"Unsupported preferred SSG `{preferred}`.",
                )

        raw_candidates = ssg.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            add_error("invalid_ssg_candidates", "$.ssg.candidates", "Must be a non-empty array.")
        else:
            for idx, item in enumerate(raw_candidates):
                if not isinstance(item, str) or not item.strip():
                    add_error("invalid_ssg_candidate", f"$.ssg.candidates[{idx}]", "Must be a non-empty string.")
                    continue
                candidate = item.strip().lower()
                candidates.append(candidate)
                if candidate not in supported_ssgs:
                    add_error(
                        "unsupported_ssg_candidate",
                        f"$.ssg.candidates[{idx}]",
                        f"Unsupported SSG candidate `{candidate}`.",
                    )
            if preferred is not None and preferred not in candidates:
                add_error(
                    "preferred_not_in_candidates",
                    "$.ssg",
                    "Preferred SSG must be listed in ssg.candidates.",
                )

    pages = spec.get("pages")
    if not isinstance(pages, list) or not pages:
        add_error("invalid_pages", "$.pages", "Field `pages` must be a non-empty array.")
    else:
        seen_slugs: set[str] = set()
        empty_slug_count = 0
        for idx, page in enumerate(pages):
            page_path = f"$.pages[{idx}]"
            if not isinstance(page, dict):
                add_error("invalid_page", page_path, "Each page must be an object.")
                continue

            title = page.get("title")
            if not isinstance(title, str) or not title.strip():
                add_error("invalid_page_title", f"{page_path}.title", "Must be a non-empty string.")

            placeholder = page.get("placeholder")
            if not isinstance(placeholder, str) or not placeholder.strip():
                add_error("invalid_page_placeholder", f"{page_path}.placeholder", "Must be a non-empty string.")

            slug_raw = page.get("slug")
            if not isinstance(slug_raw, str):
                add_error("invalid_page_slug", f"{page_path}.slug", "Must be a string.")
                continue
            slug = slug_raw.strip().strip("/")
            if slug == "":
                empty_slug_count += 1
            if slug in seen_slugs:
                add_error("duplicate_page_slug", f"{page_path}.slug", f"Duplicate page slug `{slug}`.")
            seen_slugs.add(slug)

        if empty_slug_count > 1:
            add_error("multiple_home_pages", "$.pages", "Only one page may use an empty slug.")

    nav = spec.get("nav")
    if nav is not None:
        if not isinstance(nav, list):
            add_error("invalid_nav", "$.nav", "Field `nav` must be an array when provided.")
        else:
            for idx, item in enumerate(nav):
                nav_path = f"$.nav[{idx}]"
                if not isinstance(item, dict):
                    add_error("invalid_nav_item", nav_path, "Nav item must be an object.")
                    continue
                label = item.get("label")
                url = item.get("url")
                if not isinstance(label, str) or not label.strip():
                    add_error("invalid_nav_label", f"{nav_path}.label", "Must be a non-empty string.")
                if not isinstance(url, str) or not url.strip():
                    add_error("invalid_nav_url", f"{nav_path}.url", "Must be a non-empty string.")

    return _validation_result(errors)


def _validation_result(errors: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema_validation_pass": len(errors) == 0,
        "schema_error_count": len(errors),
        "schema_error_types": sorted({e["code"] for e in errors}),
        "errors": errors,
    }
