import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments } from "@experiment/db";

export default class Update extends Command {
  static description = "Update an experiment's fields";

  static args = {
    id: Args.string({ description: "Experiment ID", required: true }),
  };

  static flags = {
    status: Flags.string({ description: "New status: open, active, proven, supported, killed" }),
    priority: Flags.integer({ description: "New priority" }),
    platform: Flags.string({ description: "Platform: local, local-apple, runpod-flash" }),
    dir: Flags.string({ description: "Experiment directory path" }),
    notes: Flags.string({ description: "Replace notes" }),
    title: Flags.string({ description: "Replace title" }),
  };

  async run() {
    const { args, flags } = await this.parse(Update);

    const existing = db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!existing) {
      this.error(`Experiment "${args.id}" not found`);
    }

    const updates: Record<string, unknown> = {
      updatedAt: new Date().toISOString().slice(0, 10),
    };

    if (flags.status) updates.status = flags.status;
    if (flags.priority !== undefined) updates.priority = flags.priority;
    if (flags.platform) updates.platform = flags.platform;
    if (flags.dir) updates.experimentDir = flags.dir;
    if (flags.notes) updates.notes = flags.notes;
    if (flags.title) updates.title = flags.title;

    db.update(experiments).set(updates).where(eq(experiments.id, args.id)).run();

    this.log(`Updated ${args.id}: ${Object.keys(updates).filter((k) => k !== "updatedAt").join(", ")}`);
  }
}
