import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, evidence } from "@experiment/db";

export default class Evidence extends Command {
  static description = "Add evidence to an experiment";

  static args = {
    id: Args.string({ description: "Experiment ID", required: true }),
  };

  static flags = {
    claim: Flags.string({ description: "Evidence claim", required: true }),
    source: Flags.string({ description: "Source (file path, URL, etc.)", required: true }),
    verdict: Flags.string({ description: "Verdict: pass, fail, inconclusive" }),
    date: Flags.string({ description: "Date (ISO format, defaults to today)" }),
  };

  async run() {
    const { args, flags } = await this.parse(Evidence);

    const exp = db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!exp) {
      this.error(`Experiment "${args.id}" not found`);
    }

    const date = flags.date ?? new Date().toISOString().slice(0, 10);

    db.insert(evidence)
      .values({
        experimentId: args.id,
        date,
        claim: flags.claim,
        source: flags.source,
        verdict: (flags.verdict as any) ?? null,
      })
      .run();

    this.log(`Added evidence to ${args.id}: ${flags.claim.slice(0, 60)}...`);
  }
}
