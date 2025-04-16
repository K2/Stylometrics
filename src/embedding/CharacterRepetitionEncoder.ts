export interface RepetitionEncodingOptions {
  minRepetitions: number;
  maxRepetitions: number;
  preferredCharacters: string[];
  randomizationFactor: number; // 0-1 scale of how much to randomize repetition counts
}

export class CharacterRepetitionEncoder {
  private options: RepetitionEncodingOptions;
  
  constructor(options?: Partial<RepetitionEncodingOptions>) {
    this.options = {
      minRepetitions: options?.minRepetitions || 2,
      maxRepetitions: options?.maxRepetitions || 7,
      preferredCharacters: options?.preferredCharacters || ['a', 'e', 'i', 'o', 'u', 'n', 's', 'l'],
      randomizationFactor: options?.randomizationFactor || 0.2
    };
  }
  
  /**
   * Finds suitable words for character repetition encoding in a text
   */
  public findEncodableCandidates(text: string): Array<{word: string, index: number, char: string, maxBits: number}> {
    const words = text.split(/\s+/);
    const candidates = [];
    
    let currentIndex = 0;
    for (const word of words) {
      // Find words with encodable characters
      for (const char of this.options.preferredCharacters) {
        if (word.includes(char)) {
          // Calculate maximum bits we can encode
          const maxRepetitions = this.options.maxRepetitions;
          const minRepetitions = this.options.minRepetitions;
          const possibleStates = maxRepetitions - minRepetitions + 1;
          const maxBits = Math.floor(Math.log2(possibleStates));
          
          if (maxBits > 0) {
            candidates.push({
              word,
              index: currentIndex, 
              char,
              maxBits
            });
            break; // Only use the first encodable character in each word
          }
        }
      }
      
      currentIndex += word.length + 1; // +1 for the space
    }
    
    return candidates;
  }
  
  /**
   * Encodes bits by repeating characters in words
   */
  public encode(text: string, bits: boolean[]): string {
    const candidates = this.findEncodableCandidates(text);
    if (candidates.length === 0) {
      throw new Error("No suitable encoding candidates found in text");
    }
    
    let bitsUsed = 0;
    const result = text.split(/\s+/);
    
    // Distribute bits across available candidates
    for (const candidate of candidates) {
      if (bitsUsed >= bits.length) break;
      
      const bitsToEncode = Math.min(candidate.maxBits, bits.length - bitsUsed);
      const valueToBinary = bits.slice(bitsUsed, bitsUsed + bitsToEncode)
        .map(b => b ? '1' : '0')
        .join('');
      const value = parseInt(valueToBinary, 2);
      
      // Convert binary value to repetition count
      const repetitions = this.options.minRepetitions + value;
      
      // Apply randomization to make patterns less detectable
      const actualRepetitions = this.applyRandomization(repetitions);
      
      // Replace the word with the encoded version
      const wordIndex = result.findIndex(w => w === candidate.word);
      if (wordIndex >= 0) {
        const charIndex = candidate.word.indexOf(candidate.char);
        result[wordIndex] = 
          candidate.word.substring(0, charIndex + 1) + 
          candidate.char.repeat(actualRepetitions - 1) + 
          candidate.word.substring(charIndex + 1);
      }
      
      bitsUsed += bitsToEncode;
    }
    
    return result.join(' ');
  }
  
  /**
   * Extracts bits encoded through character repetition
   */
  public decode(text: string, originalText: string): boolean[] {
    const originalWords = originalText.split(/\s+/);
    const encodedWords = text.split(/\s+/);
    
    const extractedBits: boolean[] = [];
    
    // Match words and detect repetition patterns
    for (let i = 0; i < originalWords.length && i < encodedWords.length; i++) {
      const originalWord = originalWords[i];
      const encodedWord = encodedWords[i];
      
      for (const char of this.options.preferredCharacters) {
        if (originalWord.includes(char)) {
          const originalCount = this.countConsecutive(originalWord, char);
          const encodedCount = this.countConsecutive(encodedWord, char);
          
          if (encodedCount > originalCount) {
            // Found an encoded character
            const repeats = encodedCount - this.options.minRepetitions;
            
            // Convert to bits
            const bitCount = Math.floor(Math.log2(this.options.maxRepetitions - this.options.minRepetitions + 1));
            const binaryValue = repeats.toString(2).padStart(bitCount, '0');
            
            // Add extracted bits
            for (let j = 0; j < binaryValue.length; j++) {
              extractedBits.push(binaryValue[j] === '1');
            }
            
            break; // Only use first encodable character
          }
        }
      }
    }
    
    return extractedBits;
  }
  
  private countConsecutive(word: string, char: string): number {
    const matches = word.match(new RegExp(`${char}+`, 'g'));
    if (!matches) return 0;
    
    return Math.max(...matches.map(m => m.length));
  }
  
  private applyRandomization(repetitions: number): number {
    if (this.options.randomizationFactor === 0) return repetitions;
    
    const factor = this.options.randomizationFactor;
    const randomOffset = Math.random() * factor * 2 - factor;
    return Math.max(this.options.minRepetitions, Math.min(
      this.options.maxRepetitions,
      Math.round(repetitions * (1 + randomOffset))
    ));
  }
}