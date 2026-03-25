import { createClient, type Client } from "@libsql/client";
import { drizzle, type LibSQLDatabase } from "drizzle-orm/libsql";
import * as schema from "./schema";

let _client: Client | null = null;
let _db: LibSQLDatabase<typeof schema> | null = null;

function getClient(): Client {
  if (!_client) {
    const url = process.env.TURSO_DATABASE_URL;
    const authToken = process.env.TURSO_AUTH_TOKEN;
    if (!url) throw new Error("TURSO_DATABASE_URL is not set — add it to .env");
    _client = createClient({ url, authToken });
  }
  return _client;
}

export function getDb(): LibSQLDatabase<typeof schema> {
  if (!_db) {
    _db = drizzle(getClient(), { schema });
  }
  return _db;
}

// Convenience: named export that matches the old `db` usage via a proxy
export const db = new Proxy({} as LibSQLDatabase<typeof schema>, {
  get(_target, prop) {
    return (getDb() as any)[prop];
  },
});

export type Db = LibSQLDatabase<typeof schema>;

// Run FTS + trigger setup (idempotent)
export async function initFts() {
  await getClient().executeMultiple(`
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
