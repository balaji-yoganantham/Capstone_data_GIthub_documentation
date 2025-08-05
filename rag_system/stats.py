import statistics

# Function to calculate conversation statistics
def get_conversation_stats(conversation_history, confidence_scores, response_times, memory):
    # Check if there's any conversation history
    if not conversation_history:
        return {}
    
    # Calculate and return various statistics
    return {
        "total_questions": len(conversation_history),  # Count total number of questions asked
        "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,  # Average confidence score
        "avg_response_time": statistics.mean(response_times) if response_times else 0,  # Average response time
        "memory_size": len(memory.chat_memory.messages),  # Number of messages in memory
        "unique_sources": len(set([  
            source for conv in conversation_history 
            for source in conv.get('sources', [])
        ]))
    }

# Function to get a summary of conversation memory
def get_memory_summary(memory):
    # Check if memory exists and has messages
    if not memory or not memory.chat_memory.messages:
        return "No conversation history"
    
    # Get all messages from memory
    messages = memory.chat_memory.messages
    
    # Return summary with message count and recent topics
    return f"Conversation contains {len(messages)} messages. Recent topics discussed: {', '.join([msg.content[:50] + '...' for msg in messages[-3:] if hasattr(msg, 'content')])}" 