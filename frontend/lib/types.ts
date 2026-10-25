/** The slice of OpenRouter's model object the UI reads. */
export type AsrModel = {
  id: string;
  name: string;
  description: string;
  hugging_face_id: string | null;
  languages: string[];
  loaded: boolean;
};

export type Segment = { id: number; start: number; end: number; text: string };
export type Word = { word: string; start: number; end: number };

/** OpenRouter's STT response. `segments`/`words` come with verbose_json. */
export type TranscriptionResponse = {
  text: string;
  usage: { seconds: number; cost: number };
  language?: string;
  duration?: number;
  segments?: Segment[];
  words?: Word[];
};

export type Health = {
  status: "ok";
  default_model: string;
  loaded_models: string[];
  busy: boolean;
  queue_length: number;
};

export type TranscriptionResult = TranscriptionResponse & {
  id: string;
  model: string;
  source: string;
  processingMs: number | null;
  createdAt: number;
};
