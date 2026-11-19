export interface Measurement { value: number | null; unit: string; method: string | null; confidence: number | null; frame_ref: string | null; calibration_ref: string | null }
export interface Price { amount: number | null; source: string; url: string; retrieved_at: string; converted?: boolean; condition_assumed?: string; currency?: string; listings_used?: number; original_amount?: number | null; original_currency?: string; fx_rate?: number | null }
export interface Book {
  id: string; shelf: string; position: number; frame_ref: string; status: string; detail_status?: string; suggested_title?: string; suggested_author?: string; title: string; author: string; edition: string; isbn: string; publisher: string;
  spine_height_cm: number | null; spine_thickness_cm: number | null; id_confidence: number; replacement_cost: Price; used_value: Price;
  height_measurement: Measurement; thickness_measurement: Measurement; ocr_text: string[]; flags: string[]; needs_appraisal: boolean; pricing_note: string; priced_isbn: string; owner_note: string; orientation: string;
}
export interface ItemCost { low: number | null; high: number | null; source: string; url: string; retrieved_at: string; currency?: string; listings_used?: number; sources?: { source: string; url: string; amount: number; currency: string; title?: string }[] }
export interface Item { id: string; detail_status?: string; category: string; description: string; brand_model: string; frame_ref: string; dimensions_cm: { w: number | null; h: number | null; d: number | null }; status: string; replacement_cost: ItemCost; confidence: number; material: string; dimension_method: string; dimension_confidence: number | null; flags: string[]; user_note: string }
export interface Totals { book_count: number; books_identified: number; books_unidentified: number; shelf_run_m: number | null; books_replacement_cost: number; books_used_value: number; items_replacement_cost_low: number; items_replacement_cost_high: number; excluded_from_totals: number; currency: string; books_priced_replacement: number; books_priced_used: number; items_priced: number; per_shelf: Record<string, { book_count: number; measured_thickness_count: number; run_m: number | null; run_complete: boolean }> }
export interface Room { length_m: number | null; width_m: number | null; height_m: number | null; floor_area_m2: number | null; wall_area_m2: number | null; shelved_wall_area_m2: number | null; floor_area_ft2: number | null; wall_area_ft2: number | null; shelved_wall_area_ft2: number | null; scale_method: string; confidence: number | null; shape: string; assumptions: string[]; evidence_frames: string[]; status: string }
export interface Review { ref_id: string; reason: string; category: string; severity: string; frame_ref: string; resolved: boolean }
export interface SecondRow { book_id: string; title: string; isbn: string; replacement_cost: number | null; used_value: number | null; currency: string; source: string; url: string; retrieved_at: string; note: string }
export interface Packet {
  sweep: { id: string; captured_at: string; device: string; duration_s: number | null; country: string; currency: string };
  room: Room; books: Book[]; items: Item[]; totals: Totals; review_queue: Review[];
  second_country_pricing: { country: string; currency: string; rows: SecondRow[] } | null;
  calibrations: { id: string; scope: string; method: string; confidence: number; frame_ref: string; note: string }[];
  processing: { stage: string; seconds: number; status: string }[]; evidence_frames: { id: string; t_s: number }[];
  run_stats?: Record<string, any>;
}
export interface LiveState { books: number; items: number; shelves: number; identified: number; elapsed_s: number; calibrated_segments: number; quality: string[]; dropped_frames: number; room: { floor_corners: number; ceiling_marks: number; camera_height_cm: number | null } }
export interface FeedEntry { id: number; kind: "event" | "guidance" | "agent" | "user" | "correction" | "processing"; text: string; t: number }
