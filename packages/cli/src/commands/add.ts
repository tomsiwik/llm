import { Command, Args, Flags } from "@oclif/core";
import { eq } from "drizzle-orm";
import { db, experiments, experimentReferences, references } from "@experiment/db";

export default class Add extends Command {
  static description = "Add a new experiment";

  static args = {
    id: Args.string({ description: "Experiment ID (e.g. exp_my_experiment)", required: true }),
  };

  static flags = {
    title: Flags.string({ description: "Experiment title", required: true }),
    scale: Flags.string({ description: "Scale: micro or macro", default: "micro" }),
    priority: Flags.integer({ description: "Priority (1=highest)", default: 3 }),
    platform: Flags.string({ description: "Platform: local, local-apple, runpod-flash" }),
    dir: Flags.string({ description: "Experiment directory path" }),
    notes: Flags.string({ description: "Notes" }),
    "grounded-by": Flags.string({ description: "arXiv ID of grounding reference" }),
  };

  async run() {
    const { args, flags } = await this.parse(Add);

    const existing = db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (existing) {
      this.error(`Experiment "${args.id}" already exists`);
    }

    const now = new Date().toISOString().slice(0, 10);

    db.insert(experiments)
      .values({
        id: args.id,
        title: flags.title,
        status: "open",
        scale: flags.scale as any,
        priority: flags.priority,
        experimentDir: flags.dir ?? null,
        platform: (flags.platform as any) ?? null,
        notes: flags.notes ?? null,
        createdAt: now,
        updatedAt: now,
      })
      .run();

    // Link grounding reference if provided
    if (flags["grounded-by"]) {
      const ref = db
        .select()
        .from(references)
        .where(eq(references.arxivId, flags["grounded-by"]))
        .get();
      if (ref) {
        db.insert(experimentReferences)
          .values({ experimentId: args.id, referenceId: ref.id })
          .onConflictDoNothing()
          .run();
        this.log(`  Linked to reference: ${ref.title}`);
      } else {
        this.warn(`Reference with arxiv_id "${flags["grounded-by"]}" not found. Add it with: experiment ref-add`);
      }
    }

    this.log(`Created experiment: ${args.id}`);
  }
}
