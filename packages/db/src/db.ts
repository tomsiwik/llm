import { createClient } from "@libsql/client";
import { drizzle } from "drizzle-orm/libsql";
import * as schema from "./schema";

const url = process.env.TURSO_DATABASE_URL;
const authToken = process.env.TURSO_AUTH_TOKEN;

if (!url) throw new Error("TURSO_DATABASE_URL is not set");

const client = createClient({ url, authToken });

export const db = drizzle(client, { schema });
export type Db = typeof db;

// Run FTS + trigger setup (idempotent)
export async function initFts() {
  await client.executeMultiple(`
    CREATE VIRTUAL TABLE IF NOT EXISTS experiments_fts
      USING fts5(id, title, notes, content=experiments, content_rowid=rowid);

    CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts
      USING fts5(claim, source, content=evidence, content_rowid=rowid);

    CREATE TRIGGER IF NOT EXISTS experiments_ai AFTER INSERT ON experiments BEGIN
      INSERT INTO experiments_fts(rowid, id, title, notes) VALUES (new.rowid, new.id, new.title, new.notes);
    END;

    CREATE TRIGGER IF NOT EXISTS experiments_au AFTER UPDATE ON experiments BEGIN
      INSERT INTO experiments_fts(experiments_fts, rowid, id, title, notes) VALUES ('delete', old.rowid, old.id, old.title, old.notes);
      INSERT INTO experiments_fts(rowid, id, title, notes) VALUES (new.rowid, new.id, new.title, new.notes);
    END;

    CREATE TRIGGER IF NOT EXISTS evidence_ai AFTER INSERT ON evidence BEGIN
      INSERT INTO evidence_fts(rowid, claim, source) VALUES (new.rowid, new.claim, new.source);
    END;

    CREATE TRIGGER IF NOT EXISTS evidence_au AFTER UPDATE ON evidence BEGIN
      INSERT INTO evidence_fts(evidence_fts, rowid, claim, source) VALUES ('delete', old.rowid, old.claim, old.source);
      INSERT INTO evidence_fts(rowid, claim, source) VALUES (new.rowid, new.claim, new.source);
    END;
  `);
}
