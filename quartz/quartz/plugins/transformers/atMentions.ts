import { QuartzTransformerPlugin } from "../types"
import { slugTag } from "../../util/path"

function stripCode(markdown: string) {
  // remove fenced code blocks
  let s = markdown.replace(/```[\s\S]*?```/g, "")
  // remove inline code
  s = s.replace(/`[^`]*`/g, "")
  return s
}

function extractAtTokens(markdown: string): string[] {
  const src = stripCode(markdown)
  // Match @tokens that start with a letter and then contain letters/numbers/_/- or slashes for hierarchy
  const re = /(^|[^\w/])@([A-Za-z][\w/-]*)/g
  const out: string[] = []
  for (const match of src.matchAll(re)) {
    const raw = match[2]
    if (!raw) continue
    out.push(slugTag(raw))
  }
  return [...new Set(out)]
}

export const AtMentions: QuartzTransformerPlugin = () => {
  return {
    name: "AtMentions",
    markdownPlugins() {
      return [
        () => {
          return (_, file) => {
            const raw = Buffer.from(file.value as Uint8Array).toString("utf8")
            file.data.atMentions = extractAtTokens(raw)
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    atMentions: string[]
  }
}
