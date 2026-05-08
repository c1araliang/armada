import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import BodyConstructor from "../../components/Body"
import { pageResources, renderPage } from "../../components/renderPage"
import { ProcessedContent, QuartzPluginData, defaultProcessedContent } from "../vfile"
import { FullPageLayout } from "../../cfg"
import { FullSlug, getAllSegmentPrefixes, joinSegments, pathToRoot } from "../../util/path"
import { defaultListPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { write } from "./helpers"
import { BuildCtx } from "../../util/ctx"
import { StaticResources } from "../../util/resources"
import AtContent from "../../components/pages/AtContent"

interface AtPageOptions extends FullPageLayout {
  sort?: (f1: QuartzPluginData, f2: QuartzPluginData) => number
}

function computeAtInfo(
  allFiles: QuartzPluginData[],
  content: ProcessedContent[],
): [Set<string>, Record<string, ProcessedContent>] {
  const ats: Set<string> = new Set(
    allFiles.flatMap((data) => data.atMentions ?? []).flatMap(getAllSegmentPrefixes),
  )

  // add base index
  ats.add("index")

  const atDescriptions: Record<string, ProcessedContent> = Object.fromEntries(
    [...ats].map((at) => {
      const title = at === "index" ? "@ Index" : `@${at}`
      return [
        at,
        defaultProcessedContent({
          slug: joinSegments("ats", at) as FullSlug,
          frontmatter: { title, tags: [] },
        }),
      ]
    }),
  )

  // Update with actual content if available (content/ats/<name>.md)
  for (const [tree, file] of content) {
    const slug = file.data.slug!
    if (slug.startsWith("ats/")) {
      const at = slug.slice("ats/".length)
      if (ats.has(at)) {
        atDescriptions[at] = [tree, file]
      }
    }
  }

  return [ats, atDescriptions]
}

async function processAtPage(
  ctx: BuildCtx,
  at: string,
  atContent: ProcessedContent,
  allFiles: QuartzPluginData[],
  opts: FullPageLayout,
  resources: StaticResources,
) {
  const slug = joinSegments("ats", at) as FullSlug
  const [tree, file] = atContent
  const cfg = ctx.cfg.configuration
  const externalResources = pageResources(pathToRoot(slug), resources)
  const componentData: QuartzComponentProps = {
    ctx,
    fileData: file.data,
    externalResources,
    cfg,
    children: [],
    tree,
    allFiles,
  }

  const content = renderPage(cfg, slug, componentData, opts, externalResources)
  return write({
    ctx,
    content,
    slug: file.data.slug!,
    ext: ".html",
  })
}

export const AtPage: QuartzEmitterPlugin<Partial<AtPageOptions>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultListPageLayout,
    pageBody: AtContent({ sort: userOpts?.sort }),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, left, right, footer: Footer } = opts
  const Header = HeaderConstructor()
  const Body = BodyConstructor()

  return {
    name: "AtPage",
    getQuartzComponents() {
      return [
        Head,
        Header,
        Body,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...left,
        ...right,
        Footer,
      ]
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map((c) => c[1].data)
      const [ats, atDescriptions] = computeAtInfo(allFiles, content)

      for (const at of ats) {
        yield processAtPage(ctx, at, atDescriptions[at], allFiles, opts, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map((c) => c[1].data)

      const affected: Set<string> = new Set()
      for (const changeEvent of changeEvents) {
        if (!changeEvent.file) continue
        const slug = changeEvent.file.data.slug!

        if (slug.startsWith("ats/")) {
          affected.add(slug.slice("ats/".length))
        }

        const fileAts = changeEvent.file.data.atMentions ?? []
        fileAts.flatMap(getAllSegmentPrefixes).forEach((at) => affected.add(at))

        affected.add("index")
      }

      if (affected.size > 0) {
        const [_ats, atDescriptions] = computeAtInfo(allFiles, content)
        for (const at of affected) {
          if (atDescriptions[at]) {
            yield processAtPage(ctx, at, atDescriptions[at], allFiles, opts, resources)
          }
        }
      }
    },
  }
}
