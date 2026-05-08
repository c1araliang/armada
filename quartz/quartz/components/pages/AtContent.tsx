import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import style from "../styles/listPage.scss"
import { PageList, SortFn } from "../PageList"
import { FullSlug, getAllSegmentPrefixes, resolveRelative, simplifySlug } from "../../util/path"
import { QuartzPluginData } from "../../plugins/vfile"
import { Root } from "hast"
import { htmlToJsx } from "../../util/jsx"
import { ComponentChildren } from "preact"
import { concatenateResources } from "../../util/resources"

interface AtContentOptions {
  sort?: SortFn
  numPages: number
}

const defaultOptions: AtContentOptions = {
  numPages: 10,
}

export default ((opts?: Partial<AtContentOptions>) => {
  const options: AtContentOptions = { ...defaultOptions, ...opts }

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
      const atItemMap: Map<string, QuartzPluginData[]> = new Map()
      for (const token of ats) {
        atItemMap.set(token, allPagesWithAt(token))
      }

      return (
        <div class="popover-hint">
          <article class={classes}>
            <p>{content}</p>
          </article>
          <p>Total @{ats.length}</p>
          <div>
            {ats.map((token) => {
              const pages = atItemMap.get(token)!
              const listProps = {
                ...props,
                allFiles: pages,
              }

              const contentPage = allFiles.filter((file) => file.slug === `ats/${token}`).at(0)
              const root = contentPage?.htmlAst
              const desc =
                !root || root?.children.length === 0
                  ? contentPage?.description
                  : htmlToJsx(contentPage.filePath!, root)

              const listingPage = `/ats/${token}` as FullSlug
              const href = resolveRelative(fileData.slug!, listingPage)

              return (
                <div>
                  <h2>
                    <a class="internal tag-link" href={href}>
                      @{token}
                    </a>
                  </h2>
                  {desc && <p>{desc}</p>}
                  <div class="page-listing">
                    <p>
                      Items: {pages.length}
                      {pages.length > options.numPages && (
                        <>
                          {" "}
                          <span>Showing first {options.numPages}</span>
                        </>
                      )}
                    </p>
                    <PageList limit={options.numPages} {...listProps} sort={options?.sort} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    } else {
      const pages = allPagesWithAt(at)
      const listProps = {
        ...props,
        allFiles: pages,
      }

      return (
        <div class="popover-hint">
          <article class={classes}>{content}</article>
          <div class="page-listing">
            <p>
              Items under @{at}: {pages.length}
            </p>
            <div>
              <PageList {...listProps} sort={options?.sort} />
            </div>
          </div>
        </div>
      )
    }
  }

  AtContent.css = concatenateResources(style, PageList.css)
  return AtContent
}) satisfies QuartzComponentConstructor
