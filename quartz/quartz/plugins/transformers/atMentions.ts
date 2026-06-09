import { QuartzTransformerPlugin } from "../types"
import { slugTag } from "../../util/path"

function splitByFences(src: string) {
  // Keep the ``` fence blocks intact and only process outside them
  return src.split(/(```[\s\S]*?```)/g)
}

function linkifyAtMentions(markdown: string) {
  const chunks = splitByFences(markdown)
  return chunks
    .map((chunk) => {
      if (chunk.startsWith("```")) return chunk

      // Avoid touching inline code spans
      const parts = chunk.split(/(`[^`]*`)/g)
      return parts
        .map((p) => {
          if (p.startsWith("`") && p.endsWith("`")) return p

          // Don't double-link if user already wrote a wikilink to ats/
          // Replace @token with [[ats/token|@token]]
          return p.replace(/(^|[^\w/])@([A-Za-z][\w/-]*)/g, (_m, pre, raw) => {
            const token = slugTag(raw)
            const display = `@${raw}`
            const link = `[[ats/${token}|${display}]]`
            return `${pre}${link}`
          })
        })
        .join("")
    })
    .join("")
}

function stripCode(markdown: string) {
  // remove fenced code blocks
  let s = markdown.replace(/```[\s\S]*?```/g, "")
  // remove inline code
  s = s.replace(/`[^`]*`/g, "")
  return s
}

function normalizeAtLine(rawLine: string, token: string) {
  let line = rawLine
  // drop common daily-log date prefix
  line = line.replace(/^\s*\d{4}-\d{2}-\d{2}:\s*/, "")
  // if @tokens were already linkified into [[ats/...|@token]], unwrap back to @token
  line = line.replace(/\[\[ats\/[^\]|]+?\|(@[^\]]+)\]\]/g, "$1")
  // remove the first occurrence of the token
  line = line.replace(new RegExp(String.raw`(^|\\s)@${token}(\\s|$)`), " ")
  // normalize whitespace
  line = line.replace(/\s+/g, " ").trim()
  return line
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

function extractAtLines(markdown: string): Record<string, string[]> {
  const src = stripCode(markdown)
  const tokens = extractAtTokens(src)
  const tokenSet = new Set(tokens)
  const lines = src.split("\n")

  const out: Record<string, string[]> = Object.fromEntries(tokens.map((t) => [t, []]))

  // capture each line under each token that appears in it
  const re = /(^|[^\w/])@([A-Za-z][\w/-]*)/g
  for (const rawLine of lines) {
    const found = new Set<string>()
    for (const match of rawLine.matchAll(re)) {
      const raw = match[2]
      if (!raw) continue
      const token = slugTag(raw)
      if (tokenSet.has(token)) found.add(token)
    }
    for (const token of found) {
      const cleaned = normalizeAtLine(rawLine, token)
      if (cleaned.length > 0) out[token].push(cleaned)
    }
  }

  // de-dupe per token
  for (const token of Object.keys(out)) {
    out[token] = [...new Set(out[token])]
  }

  return out
}

export const AtMentions: QuartzTransformerPlugin = () => {
  return {
    name: "AtMentions",
    textTransform(_ctx, src) {
      return linkifyAtMentions(src)
    },
    markdownPlugins() {
      return [
        () => {
          return (_, file) => {
            const raw = Buffer.from(file.value as Uint8Array).toString("utf8")
            file.data.atMentions = extractAtTokens(raw)
            file.data.atMentionLines = extractAtLines(raw)
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    atMentions: string[]
    atMentionLines: Record<string, string[]>
  }
}
