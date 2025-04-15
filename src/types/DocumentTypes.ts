/**
 * DocumentTypes.ts
 * 
 * This file defines the document structure types used by the carrier matrix system.
 * It models documents as hierarchical structures with segments that can be processed
 * independently for steganographic encoding.
 * 
 * Design Goals:
 * - Model document structure in a way that supports hierarchical analysis
 * - Enable segment-specific capacity and encoding decisions
 * - Support various document formats and structures
 * - Facilitate document traversal and manipulation
 */

/**
 * Document types for the stylometric analysis system
 */

interface DocumentMetadata {
  /** Author information */
  author?: string;
  
  /** Document title */
  title?: string;
  
  /** Date information */
  date?: Date | string;
  
  /** Publication date */
  publicationDate?: Date;
  
  /** Document source */
  source?: string;
  
  /** Language of document content */
  language?: string;
  
  /** Document version or edition */
  version?: string;
  
  /** Genre or document category */
  genre?: string;
  
  /** Any custom properties for specific encoding techniques */
  customProperties?: Record<string, any>;
  
  [key: string]: any; // Allow for additional metadata fields
}

interface Document {
  /** Unique identifier for this document */
  id: string;
  
  /** Document content */
  content: string;
  
  /** Document title */
  title?: string;
  
  /** Document segments in hierarchical organization */
  segments?: DocumentSegment[];
  
  /** Original document format */
  format?: DocumentFormat;
  
  /** Document metadata for processing decisions */
  metadata?: DocumentMetadata;
}

interface ProcessedDocument extends Document {
  tokens?: string[];
  sentences?: string[];
  features?: Record<string, number | string | boolean | Array<any>>;
}

/**
 * Represents a segment of a document (chapter, section, etc.)
 * Each segment can be processed independently for steganography
 */
interface DocumentSegment {
  /** Unique identifier for this segment */
  id: string;
  
  /** Raw content of the segment */
  content: string;
  
  /** Type of segment, indicating its role in the document */
  type: SegmentType;
  
  /** Hierarchical level (1 = top level, higher numbers are deeper) */
  level: number;
  
  /** Parent segment ID for hierarchical structures */
  parentId?: string;
  
  /** Custom metadata for segment-specific processing */
  metadata?: Record<string, any>;
  
  /** Position in the document for reassembly */
  position: number;
}

/**
 * Document metadata containing information useful for steganographic processing
 */
interface DocumentMetadata {
  /** Author information */
  author?: string;
  
  /** Publication date */
  publicationDate?: Date;
  
  /** Document version or edition */
  version?: string;
  
  /** Language of document content */
  language?: string;
  
  /** Genre or document category */
  genre?: string;
  
  /** Any custom properties for specific encoding techniques */
  customProperties?: Record<string, any>;
}

/**
 * Supported document formats
 */
enum DocumentFormat {
  MARKDOWN = 'markdown',
  HTML = 'html',
  PLAIN_TEXT = 'plain_text',
  PDF = 'pdf',
  DOCX = 'docx',
  LATEX = 'latex',
  XML = 'xml',
  JSON = 'json'
}

/**
 * Result of document analysis 
 */
interface DocumentAnalysis {
  /** Original document */
  document: Document;
  
  /** Document structure metrics */
  metrics: DocumentMetrics;
  
  /** Total steganographic capacity in bits */
  totalCapacityBits: number;
  
  /** Per-segment capacity information */
  segmentCapacities: Map<string, number>;
}

/**
 * Document structure metrics
 */
interface DocumentMetrics {
  /** Total segment count */
  segmentCount: number;
  
  /** Number of segments by type */
  segmentsByType: Record<SegmentType, number>;
  
  /** Average segment length in characters */
  averageSegmentLength: number;
  
  /** Document complexity score (higher = more complex) */
  structuralComplexity: number;
  
  /** Maximum nesting depth */
  maxDepth: number;
}

/**
 * Document parser options
 */
interface DocumentParserOptions {
  /** Parser should respect content breaks */
  preserveWhitespace?: boolean;
  
  /** Maximum segment size to split large segments */
  maxSegmentSize?: number;
  
  /** Minimum segment size to avoid tiny segments */
  minSegmentSize?: number;
  
  /** Custom rules for segment detection */
  segmentRules?: RegExp[];
}

/**
 * Represents the capacity of a document segment to carry steganographic data
 */
interface CarrierCapacity {
  /** Document segment reference */
  segmentId: string;
  
  /** Maximum capacity in bits */
  capacityBits: number;
  
  /** Encoding techniques applicable to this segment */
  applicableTechniques: EncodingTechnique[];
  
  /** Weighted risk factor (0-1, where 0 is safest) */
  detectionRisk: number;
}

/**
 * Available encoding techniques for steganography
 */
enum EncodingTechnique {
  WHITESPACE_VARIATION = 'whitespace_variation',
  SYNONYM_SUBSTITUTION = 'synonym_substitution',
  PUNCTUATION_MODIFICATION = 'punctuation_modification',
  SYNTACTIC_TRANSFORMATION = 'syntactic_transformation',
  TYPOGRAPHIC_VARIATION = 'typographic_variation',
  CHARACTER_ENCODING = 'character_encoding',
  FORMATTING_CHANGES = 'formatting_changes',
  METADATA_EMBEDDING = 'metadata_embedding',
  STRUCTURAL_MODIFICATION = 'structural_modification'
}

/**
 * Types of document segments that can be processed independently
 */
enum SegmentType {
  HEADING = 'heading',
  PARAGRAPH = 'paragraph',
  LIST = 'list',
  TABLE = 'table',
  CODE_BLOCK = 'code_block',
  BLOCKQUOTE = 'blockquote',
  IMAGE = 'image',
  FOOTNOTE = 'footnote',
  SECTION = 'section',
  CHAPTER = 'chapter',
  METADATA = 'metadata'
}
// Export all types and enums
module.exports = {
  DocumentFormat,
  EncodingTechnique,
  SegmentType
};

// TypeScript type exports
// These types don't exist at runtime but are available for TypeScript type checking
export type {
  DocumentMetadata,
  Document,
  ProcessedDocument,
  DocumentSegment,
  DocumentAnalysis,
  DocumentMetrics,
  DocumentParserOptions,
  CarrierCapacity
};
