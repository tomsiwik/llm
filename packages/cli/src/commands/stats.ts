import { Command } from "@oclif/core";
import { sql } from "drizzle-orm";
import { db, experiments, evidence, killCriteria, tags, experimentTags, references } from "@experiment/db";

export default class Stats extends Command {
  static description = "Show experiment tracking dashboard";

  async run() {
    // Status distribution
    const statusDist = db
      .select({ status: experiments.status, count: sql<number>`count(*)` })
      .from(experiments)
      .groupBy(experiments.status)
      .all();

    const total = statusDist.reduce((sum, r) => sum + r.count, 0);
    const killed = statusDist.find((r) => r.status === "killed")?.count ?? 0;
    const proven = statusDist.find((r) => r.status === "proven")?.count ?? 0;
    const supported = statusDist.find((r) => r.status === "supported")?.count ?? 0;

    this.log("\n  Experiment Stats");
    this.log("  " + "─".repeat(40));
    this.log(`  Total experiments: ${total}`);
    for (const r of statusDist.sort((a, b) => b.count - a.count)) {
      const bar = "█".repeat(Math.round((r.count / total) * 30));
      this.log(`    ${r.status.padEnd(12)} ${String(r.count).padStart(3)} ${bar}`);
    }
    this.log(`\n  Kill rate: ${((killed / total) * 100).toFixed(1)}% (${killed}/${total})`);
    this.log(`  Success rate: ${(((proven + supported) / total) * 100).toFixed(1)}% (${proven + supported}/${total})`);

    // Evidence count
    const evCount = db.select({ count: sql<number>`count(*)` }).from(evidence).get();
    this.log(`\n  Evidence entries: ${evCount?.count}`);

    // Kill criteria count
    const kcCount = db.select({ count: sql<number>`count(*)` }).from(killCriteria).get();
    this.log(`  Kill criteria: ${kcCount?.count}`);

    // References
    const refCount = db.select({ count: sql<number>`count(*)` }).from(references).get();
    this.log(`  References: ${refCount?.count}`);

    // Top tags
    const topTags = db
      .select({ name: tags.name, count: sql<number>`count(*)` })
      .from(experimentTags)
      .innerJoin(tags, sql`${experimentTags.tagId} = ${tags.id}`)
      .groupBy(tags.name)
      .orderBy(sql`count(*) DESC`)
      .limit(15)
      .all();

    this.log("\n  Top Tags:");
    for (const t of topTags) {
      this.log(`    ${t.name.padEnd(25)} ${t.count}`);
    }

    // Scale distribution
    const scaleDist = db
      .select({ scale: experiments.scale, count: sql<number>`count(*)` })
      .from(experiments)
      .groupBy(experiments.scale)
      .all();
    this.log("\n  Scale:");
    for (const s of scaleDist) {
      this.log(`    ${s.scale.padEnd(10)} ${s.count}`);
    }

    this.log("");
  }
}
