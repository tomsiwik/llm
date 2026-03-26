import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, experimentDependencies } from "@experiment/db";

export default class DepAdd extends Command {
  static description = "Add a dependency between experiments";

  static args = {
    id: Args.string({ description: "Experiment ID (the one that depends)", required: true }),
  };

  static flags = {
    on: Flags.string({ description: "Experiment ID it depends on", required: true }),
  };

  async run() {
    const { args, flags } = await this.parse(DepAdd);

    const exp = await db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!exp) this.error(`Experiment "${args.id}" not found`);

    const dep = await db.select().from(experiments).where(eq(experiments.id, flags.on)).get();
    if (!dep) this.error(`Dependency target "${flags.on}" not found`);

    await db.insert(experimentDependencies)
      .values({ experimentId: args.id, dependsOnId: flags.on })
      .onConflictDoNothing()
      .run();

    this.log(`${args.id} now depends on ${flags.on}`);
  }
}
