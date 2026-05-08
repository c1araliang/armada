import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import style from "../styles/listPage.scss"
import { FullSlug, getAllSegmentPrefixes, resolveRelative, simplifySlug } from "../../util/path"
import { QuartzPluginData } from "../../plugins/vfile"
import { Root } from "hast"
import { htmlToJsx } from "../../util/jsx"
import { ComponentChildren } from "preact"
import { concatenateResources } from "../../util/resources"

interface AtContentOptions {
  numPages: number
}

const defaultOptions: AtContentOptions = {
  numPages: 10,
}

export default ((opts?: Partial<AtContentOptions>) => {
  const options: AtContentOptions = { ...defaultOptions, ...opts }

  const allAtSnippets = (file: QuartzPluginData, token: string) => {
    const direct = file.atMentionLines?.[token] ?? []
    // if token is hierarchical, include child tokens as well (same behavior as tags)
    const prefixes = (file.atMentions ?? []).flatMap(getAllSegmentPrefixes)
    const children = (file.atMentions ?? [])
      .filter((t) => t !== token && prefixes.includes(token) && t.startsWith(`${token}/`))
      .flatMap((t) => file.atMentionLines?.[t] ?? [])

    return [...new Set([...direct, ...children])]
  }

  const AtContent: QuartzComponent = (props: QuartzComponentProps) => {
    const { tree, fileData, allFiles } = props
    const slug = fileData.slug

    if (!(slug?.startsWith("ats/") || slug === "ats")) {
      throw new Error(`Component "AtContent" tried to render a non-@ page: ${slug}`)
    }

    const at = simplifySlug(slug.slice("ats/".length) as FullSlug)
    const allPagesWithAt = (atToken: string) =>
      allFiles.filter((file) =>
        (file.atMentions ?? []).flatMap(getAllSegmentPrefixes).includes(atToken),
      )

    const content = (
      (tree as Root).children.length === 0
        ? fileData.description
        : htmlToJsx(fileData.filePath!, tree)
    ) as ComponentChildren
    const cssClasses: string[] = fileData.frontmatter?.cssclasses ?? []
    const classes = cssClasses.join(" ")

    if (at === "/") {
      const ats = [
        ...new Set(
          allFiles.flatMap((data) => data.atMentions ?? []).flatMap(getAllSegmentPrefixes),
        ),
      ].sort((a, b) => a.localeCompare(b))

      return (
        <div class="popover-hint">
          <article class={classes}>
            <p>{content}</p>
          </article>
          <p>Total @{ats.length}</p>
          <div>
            {ats.map((token) => {
              const listingPage = `/ats/${token}` as FullSlug
              const href = resolveRelative(fileData.slug!, listingPage)

              const pages = allPagesWithAt(token)
              const items = pages
                .flatMap((f) =>
                  allAtSnippets(f, token).map((text) => ({
                    text,
                    slug: f.slug!,
                  })),
                )
                .slice(0, options.numPages)

              return (
                <div>
                  <h2>
                    <a class="internal tag-link" href={href}>
                      @{token}
                    </a>
                  </h2>
                  <div class="page-listing">
                    <p>
                      Items: {pages.reduce((acc, f) => acc + allAtSnippets(f, token).length, 0)}
                    </p>
                    <ul>
                      {items.map((it) => (
                        <li>
                          <a class="internal" href={resolveRelative(fileData.slug!, it.slug)}>
                            {it.text}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    } else {
      const pages = allPagesWithAt(at)
      const items = pages
        .flatMap((f) =>
          allAtSnippets(f, at).map((text) => ({
            text,
            slug: f.slug!,
          })),
        )
        .slice(0, options.numPages)

      return (
        <div class="popover-hint">
          <article class={classes}>{content}</article>
          <div class="page-listing">
            <p>
              Items under @{at}: {pages.reduce((acc, f) => acc + allAtSnippets(f, at).length, 0)}
            </p>
            <ul>
              {items.map((it) => (
                <li>
                  <a class="internal" href={resolveRelative(fileData.slug!, it.slug)}>
                    {it.text}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )
    }
  }

  AtContent.css = concatenateResources(style)
  return AtContent
}) satisfies QuartzComponentConstructor
