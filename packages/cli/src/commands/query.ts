import { Command, Args } from "@oclif/core";
import { createClient } from "@libsql/client";

export default class Query extends Command {
  static description = "Full-text search across experiments and evidence";

  static args = {
    text: Args.string({ description: "Search query", required: true }),
  };

  async run() {
    const { args } = await this.parse(Query);

    const client = createClient({
      url: process.env.TURSO_DATABASE_URL!,
      authToken: process.env.TURSO_AUTH_TOKEN,
    });

    // Search experiments
    const expResults = await client.execute({
      sql: `SELECT id, title, snippet(experiments_fts, 1, '>>>', '<<<', '...', 30) as snippet, rank
            FROM experiments_fts
            WHERE experiments_fts MATCH ?
            ORDER BY rank
            LIMIT 15`,
      args: [args.text],
    });

    // Search evidence
    const evResults = await client.execute({
      sql: `SELECT e.experiment_id, snippet(evidence_fts, 0, '>>>', '<<<', '...', 30) as snippet, evidence_fts.rank
            FROM evidence_fts
            JOIN evidence e ON e.rowid = evidence_fts.rowid
            WHERE evidence_fts MATCH ?
            ORDER BY evidence_fts.rank
            LIMIT 15`,
      args: [args.text],
    });

    if (expResults.rows.length === 0 && evResults.rows.length === 0) {
      this.log(`No results for "${args.text}"`);
      return;
    }

    if (expResults.rows.length > 0) {
      this.log(`\n  Experiments matching "${args.text}":`);
      for (const r of expResults.rows) {
        this.log(`    ${r.id}`);
        this.log(`      ${r.snippet}`);
      }
    }

    if (evResults.rows.length > 0) {
      this.log(`\n  Evidence matching "${args.text}":`);
      for (const r of evResults.rows) {
        this.log(`    [${r.experiment_id}]`);
        this.log(`      ${r.snippet}`);
      }
    }

    this.log(`\n  ${expResults.rows.length} experiments, ${evResults.rows.length} evidence entries found`);
  }
}
