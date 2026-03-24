import { Command, Args } from "@oclif/core";
import { Database } from "bun:sqlite";
import { resolve } from "path";

// Resolve db path relative to the monorepo root
function findDbPath(): string {
  // Walk up from CLI package to find packages/db/data/experiments.db
  let dir = import.meta.dirname ?? process.cwd();
  for (let i = 0; i < 5; i++) {
    const candidate = resolve(dir, "packages/db/data/experiments.db");
    if (Bun.file(candidate).size > 0) return candidate;
    dir = resolve(dir, "..");
  }
  return resolve(process.cwd(), "packages/db/data/experiments.db");
}

export default class Query extends Command {
  static description = "Full-text search across experiments and evidence";

  static args = {
    text: Args.string({ description: "Search query", required: true }),
  };

  async run() {
    const { args } = await this.parse(Query);
    const dbPath = findDbPath();

    // Use raw sqlite for FTS5 queries (drizzle doesn't support virtual tables)
    const sqlite = new Database(dbPath, { readonly: true });

    // Search experiments
    const expResults = sqlite
      .query(
        `SELECT id, title, snippet(experiments_fts, 1, '>>>', '<<<', '...', 30) as snippet, rank
         FROM experiments_fts
         WHERE experiments_fts MATCH ?
         ORDER BY rank
         LIMIT 15`,
      )
      .all(args.text) as { id: string; title: string; snippet: string; rank: number }[];

    // Search evidence
    const evResults = sqlite
      .query(
        `SELECT e.experiment_id, snippet(evidence_fts, 0, '>>>', '<<<', '...', 30) as snippet, evidence_fts.rank
         FROM evidence_fts
         JOIN evidence e ON e.rowid = evidence_fts.rowid
         WHERE evidence_fts MATCH ?
         ORDER BY evidence_fts.rank
         LIMIT 15`,
      )
      .all(args.text) as { experiment_id: string; snippet: string; rank: number }[];

    if (expResults.length === 0 && evResults.length === 0) {
      this.log(`No results for "${args.text}"`);
      return;
    }

    if (expResults.length > 0) {
      this.log(`\n  Experiments matching "${args.text}":`);
      for (const r of expResults) {
        this.log(`    ${r.id}`);
        this.log(`      ${r.snippet}`);
      }
    }

    if (evResults.length > 0) {
      this.log(`\n  Evidence matching "${args.text}":`);
      for (const r of evResults) {
        this.log(`    [${r.experiment_id}]`);
        this.log(`      ${r.snippet}`);
      }
    }

    this.log(`\n  ${expResults.length} experiments, ${evResults.length} evidence entries found`);
  }
}
