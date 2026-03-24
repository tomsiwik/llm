import { Database } from "bun:sqlite";
import { drizzle } from "drizzle-orm/bun-sqlite";
import * as schema from "./schema";
import { resolve, dirname } from "path";
import { mkdirSync, existsSync } from "fs";

const DEFAULT_DB_PATH = resolve(import.meta.dirname ?? ".", "../data/experiments.db");

export function getDb(dbPath: string = DEFAULT_DB_PATH) {
  const dir = dirname(dbPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const sqlite = new Database(dbPath);
  sqlite.exec("PRAGMA journal_mode = WAL");
  sqlite.exec("PRAGMA foreign_keys = ON");

  const db = drizzle(sqlite, { schema });

  initFts(sqlite);

  return db;
}

function initFts(sqlite: Database) {
  sqlite.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS experiments_fts
      USING fts5(id, title, notes, content=experiments, content_rowid=rowid);
  `);
  sqlite.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts
      USING fts5(claim, source, content=evidence, content_rowid=rowid);
  `);

  // Triggers to keep FTS in sync
  sqlite.exec(`
    CREATE TRIGGER IF NOT EXISTS experiments_ai AFTER INSERT ON experiments BEGIN
      INSERT INTO experiments_fts(rowid, id, title, notes) VALUES (new.rowid, new.id, new.title, new.notes);
    END;
  `);
  sqlite.exec(`
    CREATE TRIGGER IF NOT EXISTS experiments_au AFTER UPDATE ON experiments BEGIN
      INSERT INTO experiments_fts(experiments_fts, rowid, id, title, notes) VALUES ('delete', old.rowid, old.id, old.title, old.notes);
      INSERT INTO experiments_fts(rowid, id, title, notes) VALUES (new.rowid, new.id, new.title, new.notes);
    END;
  `);
  sqlite.exec(`
    CREATE TRIGGER IF NOT EXISTS evidence_ai AFTER INSERT ON evidence BEGIN
      INSERT INTO evidence_fts(rowid, claim, source) VALUES (new.rowid, new.claim, new.source);
    END;
  `);
  sqlite.exec(`
    CREATE TRIGGER IF NOT EXISTS evidence_au AFTER UPDATE ON evidence BEGIN
      INSERT INTO evidence_fts(evidence_fts, rowid, claim, source) VALUES ('delete', old.rowid, old.claim, old.source);
      INSERT INTO evidence_fts(rowid, claim, source) VALUES (new.rowid, new.claim, new.source);
    END;
  `);
}

export type Db = ReturnType<typeof getDb>;
