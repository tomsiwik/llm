import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, successCriteria } from "@experiment/db";

export default class SuccessAdd extends Command {
  static description = "Add a success criterion to an experiment";

  static args = {
    id: Args.string({ description: "Experiment ID", required: true }),
  };

  static flags = {
    condition: Flags.string({ description: "Success condition", required: true }),
    unlocks: Flags.string({ description: "What this success unlocks", required: true }),
    reason: Flags.string({ description: "Why this criterion matters", default: "pre-registered" }),
    "max-followup": Flags.integer({ description: "Max follow-up experiments", default: 1 }),
  };

  async run() {
    const { args, flags } = await this.parse(SuccessAdd);

    const exp = await db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!exp) this.error(`Experiment "${args.id}" not found`);

    const result = await db.insert(successCriteria).values({
      experimentId: args.id,
      condition: flags.condition,
      unlocks: flags.unlocks,
      maxFollowup: flags["max-followup"],
      reason: flags.reason,
    }).run();

    this.log(`Added success criterion #${result.lastInsertRowid} to ${args.id}`);
  }
}
