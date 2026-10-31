import raw from '../data/vocabulary.json';

export type WordClass = 'core' | 'verb' | 'noun' | 'emotion' | 'person' | 'social';
export type LabelLang = 'lb' | 'msa' | 'en';
export type Gender = 'm' | 'f';

interface RawChoice {
  lb: string;
  msa: string;
  en: string;
  search: string;
  arasaac: number;
}

interface RawWord {
  id: string;
  topic: string;
  class: WordClass;
  lb?: string;
  lb_alt?: string[];
  lb_f?: string;
  msa?: string;
  msa_f?: string;
  f_means?: 'speaker' | 'listener';
  en?: string;
  search?: string;
  arasaac?: number;
  note?: string;
  choices?: RawChoice[];
}

interface RawVocabulary {
  version: number;
  colours: Record<WordClass, string | null>;
  pictograms: { image_url: string; attribution: string };
  words: RawWord[];
}

/** One cell on the grid. A word with choices (brother, sister) gives one cell per choice. */
export interface Cell {
  key: string;
  wordId: string;
  topic: string;
  cls: WordClass;
  lb: string;
  lbF?: string;
  msa: string;
  msaF?: string;
  en: string;
  fMeans?: 'speaker' | 'listener';
  arasaac: number;
}

const vocabulary = raw as unknown as RawVocabulary;

export const ATTRIBUTION = vocabulary.pictograms.attribution;

export function pictogramUrl(id: number): string {
  return vocabulary.pictograms.image_url.split('{id}').join(String(id));
}

// Display shades for the colour names in vocabulary.json (the therapist's scheme).
const SHADES: Record<string, string> = {
  black: '#1f1f1f',
  red: '#d32f2f',
  blue: '#1565c0',
  orange: '#ef6c00',
  green: '#2e7d32',
};

/** Border colour for a word class; classes without a colour fall back to the core colour. */
export function classColour(cls: WordClass): string {
  const name = vocabulary.colours[cls] ?? vocabulary.colours.core ?? 'black';
  return SHADES[name] ?? name;
}

export function cells(): Cell[] {
  const out: Cell[] = [];
  for (const w of vocabulary.words) {
    if (w.choices) {
      w.choices.forEach((c, i) =>
        out.push({ key: `${w.id}-${i}`, wordId: w.id, topic: w.topic, cls: w.class, lb: c.lb, msa: c.msa, en: c.en, arasaac: c.arasaac }),
      );
    } else if (w.lb && w.msa && w.en && w.arasaac !== undefined) {
      out.push({
        key: w.id,
        wordId: w.id,
        topic: w.topic,
        cls: w.class,
        lb: w.lb,
        lbF: w.lb_f,
        msa: w.msa,
        msaF: w.msa_f,
        en: w.en,
        fMeans: w.f_means,
        arasaac: w.arasaac,
      });
    }
  }
  return out;
}

/** The label to show and speak. Feminine forms apply when they mark the speaker and the user is female. */
export function label(cell: Cell, lang: LabelLang, gender: Gender): string {
  if (lang === 'en') return cell.en;
  const feminine = gender === 'f' && cell.fMeans === 'speaker';
  if (lang === 'msa') return feminine && cell.msaF ? cell.msaF : cell.msa;
  return feminine && cell.lbF ? cell.lbF : cell.lb;
}
