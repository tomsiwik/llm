import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, killCriteria } from "@experiment/db";

export default class KillAdd extends Command {
  static description = "Add a kill criterion to an experiment";

  static args = {
    id: Args.string({ description: "Experiment ID", required: true }),
  };

  static flags = {
    text: Flags.string({ description: "Kill criterion text", required: true }),
    reason: Flags.string({ description: "Why this criterion exists", default: "pre-registered" }),
  };

  async run() {
    const { args, flags } = await this.parse(KillAdd);

    const exp = await db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!exp) this.error(`Experiment "${args.id}" not found`);

    const result = await db.insert(killCriteria).values({
      experimentId: args.id,
      text: flags.text,
      reason: flags.reason,
      result: "untested",
    }).run();

    this.log(`Added kill criterion #${result.lastInsertRowid} to ${args.id}`);
  }
}
