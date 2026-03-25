import { Command, Args } from "@oclif/core";
import { eq } from "drizzle-orm";
import {
  db,
  experiments,
  killCriteria,
  successCriteria,
  evidence,
  experimentTags,
  tags,
  experimentDependencies,
  experimentReferences,
  references,
} from "@experiment/db";

export default class Get extends Command {
  static description = "Get full details of an experiment";

  static args = {
    id: Args.string({ description: "Experiment ID", required: true }),
  };

  async run() {
    const { args } = await this.parse(Get);

    const exp = db.select().from(experiments).where(eq(experiments.id, args.id)).get();
    if (!exp) {
      this.error(`Experiment "${args.id}" not found`);
    }

    // Header
    this.log(`\n${exp.id}`);
    this.log(`  Title:    ${exp.title}`);
    this.log(`  Status:   ${exp.status}`);
    this.log(`  Scale:    ${exp.scale}  Priority: ${exp.priority}`);
    this.log(`  Platform: ${exp.platform ?? "—"}`);
    this.log(`  Dir:      ${exp.experimentDir ?? "—"}`);
    this.log(`  Created:  ${exp.createdAt}  Updated: ${exp.updatedAt}`);

    // Dependencies
    const deps = db
      .select({ id: experimentDependencies.dependsOnId })
      .from(experimentDependencies)
      .where(eq(experimentDependencies.experimentId, args.id))
      .all();
    if (deps.length > 0) {
      this.log(`\n  Depends on:`);
      for (const d of deps) this.log(`    - ${d.id}`);
    }

    // Blocks (reverse deps)
    const blocks = db
      .select({ id: experimentDependencies.experimentId })
      .from(experimentDependencies)
      .where(eq(experimentDependencies.dependsOnId, args.id))
      .all();
    if (blocks.length > 0) {
      this.log(`  Blocks:`);
      for (const b of blocks) this.log(`    - ${b.id}`);
    }

    // Kill criteria
    const kcs = db.select().from(killCriteria).where(eq(killCriteria.experimentId, args.id)).all();
    if (kcs.length > 0) {
      this.log(`\n  Kill Criteria:`);
      for (const kc of kcs) {
        const icon = kc.result === "pass" ? "✓" : kc.result === "fail" ? "✗" : kc.result === "inconclusive" ? "?" : "·";
        this.log(`    [${icon}] ${kc.text}`);
        if (kc.reason !== "pre-registered") this.log(`        Reason: ${kc.reason}`);
      }
    }

    // Success criteria
    const scs = db.select().from(successCriteria).where(eq(successCriteria.experimentId, args.id)).all();
    if (scs.length > 0) {
      this.log(`\n  Success Criteria:`);
      for (const sc of scs) {
        this.log(`    If: ${sc.condition}`);
        this.log(`    Unlocks: ${sc.unlocks} (max ${sc.maxFollowup} follow-up)`);
        this.log(`    Reason: ${sc.reason}`);
      }
    }

    // Evidence
    const evs = db.select().from(evidence).where(eq(evidence.experimentId, args.id)).all();
    if (evs.length > 0) {
      this.log(`\n  Evidence (${evs.length}):`);
      for (const ev of evs) {
        const v = ev.verdict ? ` [${ev.verdict}]` : "";
        this.log(`    ${ev.date}${v}: ${ev.claim.slice(0, 120)}`);
        if (ev.source !== "imported-from-yml") this.log(`      Source: ${ev.source}`);
      }
    }

    // Tags
    const expTags = db
      .select({ name: tags.name })
      .from(experimentTags)
      .innerJoin(tags, eq(experimentTags.tagId, tags.id))
      .where(eq(experimentTags.experimentId, args.id))
      .all();
    if (expTags.length > 0) {
      this.log(`\n  Tags: ${expTags.map((t) => t.name).join(", ")}`);
    }

    // References
    const refs = db
      .select({
        title: references.title,
        url: references.url,
        relevance: references.relevance,
        localPath: references.localPath,
      })
      .from(experimentReferences)
      .innerJoin(references, eq(experimentReferences.referenceId, references.id))
      .where(eq(experimentReferences.experimentId, args.id))
      .all();
    if (refs.length > 0) {
      this.log(`\n  References:`);
      for (const ref of refs) {
        this.log(`    ${ref.title}: ${ref.relevance.slice(0, 80)}`);
        if (ref.localPath) this.log(`      Code: ${ref.localPath}`);
      }
    }

    // Notes (truncated)
    if (exp.notes) {
      this.log(`\n  Notes: ${exp.notes.slice(0, 300)}${exp.notes.length > 300 ? "..." : ""}`);
    }

    this.log("");
  }
}
