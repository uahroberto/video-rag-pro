"""
Handles the logical grouping of transcription segments into overlapping chunks.
This preserves temporal metadata while ensuring semantic context for the RAG engine.
Estimated overlap around 15-20% so we can preserve context for the RAG engine.
"""

class ChunkingProcessor:
    def __init__(self, min_chunk_size: int = 500, overlap_segments: int = 2):
        """
        Args:
            min_chunk_size (int): Minimum character length before closing a chunk.
            overlap_segments (int): Number of segments to repeat between chunks.
        """
        self.min_chunk_size = min_chunk_size
        self.overlap_segments = overlap_segments

    def process(self, segments: list[dict]) -> list[dict]:
        """
        Groups atomic segments into larger chunks with precise start/end timestamps.
        Implements 'Structured Aggregation' as per architectural validation.
        """
        chunks = []
        current_segments = []
        current_text_len = 0

        i = 0
        while i < len(segments):
            seg = segments[i]
            current_segments.append(seg)
            current_text_len += len(seg['text'])

            # If chunk is large enough or it's the last segment, emit chunk
            if current_text_len >= self.min_chunk_size or i == len(segments) - 1:
                # The start time is the 'start' of the first segment in group
                # The end time is the 'end' of the last segment in group
                chunk_data = {
                    "text": " ".join([s['text'] for s in current_segments]).strip(),
                    "start": current_segments[0]['start'],
                    "end": current_segments[-1]['end'],
                }
                chunks.append(chunk_data)

                # Reset for next chunk, but keep overlap for context preservation (essential for RAG)
                if i < len(segments) - 1:
                    # Move back index to create overlap
                    i -= self.overlap_segments
                    if i < 0: i = 0 # Safety check
                
                current_segments = []
                current_text_len = 0
            
            i += 1

        print(f"📦 Grouped {len(segments)} segments into {len(chunks)} contextual chunks.")
        return chunks