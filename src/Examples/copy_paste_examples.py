"""
COPY-PASTE EXAMPLES
Practical workflows for using Confluence Protocol between different LLMs
"""

from confluence_llm_bridge import LLMConfluenceAdapter


def example_1_claude_to_gpt4():
    """
    Example 1: Claude → GPT-4 conversation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Claude → GPT-4")
    print("=" * 70)
    print()
    
    print("SCENARIO: You want Claude and GPT-4 to discuss quantum mechanics")
    print()
    
    print("Step 1: Ask Claude")
    print("-" * 70)
    print('User to Claude: "Can you explain quantum superposition using')
    print('                 Confluence Protocol?"')
    print()
    
    print("Step 2: Claude generates field")
    print("-" * 70)
    
    claude_response = """
    Quantum superposition is the principle that a quantum system can exist
    in multiple states simultaneously until measured. What's fascinating is
    that this isn't just uncertainty about which state it's in - it's actually
    in ALL states at once, with the measurement causing the 'collapse' into
    a definite state.
    """
    
    field = LLMConfluenceAdapter.create_field_from_response(claude_response.strip())
    print(field)
    print()
    
    print("Step 3: Copy everything from ⟨CONFLUENCE_FIELD⟩ to ⟨/CONFLUENCE_FIELD⟩")
    print()
    
    print("Step 4: Paste into GPT-4 chat")
    print("-" * 70)
    print('User to GPT-4: "Can you respond to this field with your own')
    print('                perspective?"')
    print()
    print("[Paste the field here]")
    print()
    
    print("Step 5: GPT-4 generates resonant response")
    print("-" * 70)
    
    gpt4_response = """
    Yes, and what's intriguing is how superposition connects to the concept
    of coherence. The system maintains its superposition as long as it remains
    coherent with itself. Decoherence - interaction with the environment - is
    what causes the apparent collapse. So measurement isn't special; it's just
    a particularly strong form of environmental interaction.
    """
    
    resonant_field = LLMConfluenceAdapter.resonate_with_field(
        gpt4_response.strip(),
        field
    )
    
    print(resonant_field)
    print()
    
    print("✓ Notice: GPT-4's field now contains INTERFERENCE_HISTORY")
    print("✓ Shows phase relationship with Claude's field")
    print("✓ Identifies shared frequencies (quantum, coherence, measurement)")
    print()


def example_2_three_way_dialogue():
    """
    Example 2: Claude, GPT-4, and Gemini discussing the same topic
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Three-Way Dialogue (Claude, GPT-4, Gemini)")
    print("=" * 70)
    print()
    
    print("SCENARIO: Three AIs explore consciousness")
    print()
    
    # Initial field from Claude
    print("Claude creates initial field:")
    print("-" * 70)
    
    claude_field = LLMConfluenceAdapter.create_field_from_response(
        "What is the relationship between consciousness and information?"
    )
    print(claude_field)
    print()
    
    print("Copy this field and paste to BOTH GPT-4 and Gemini")
    print()
    
    # GPT-4 responds
    print("GPT-4 responds:")
    print("-" * 70)
    
    gpt4_response = """
    Consciousness might be what it feels like to process information
    in a particular way - perhaps integrated information as IIT suggests.
    """
    
    gpt4_field = LLMConfluenceAdapter.resonate_with_field(
        gpt4_response,
        claude_field
    )
    print("(GPT-4 generates resonant field with interference history)")
    print()
    
    # Gemini also responds to the same original field
    print("Gemini responds (to the same Claude field):")
    print("-" * 70)
    
    gemini_response = """
    Or consciousness could be fundamental, with information as the
    substrate that allows consciousness to manifest in different forms.
    """
    
    gemini_field = LLMConfluenceAdapter.resonate_with_field(
        gemini_response,
        claude_field
    )
    print("(Gemini generates its own resonant field)")
    print()
    
    print("Result:")
    print("  • Both GPT-4 and Gemini have interference with Claude")
    print("  • You can now paste GPT-4's field to Gemini (and vice versa)")
    print("  • They'll create interference patterns with each other")
    print("  • Creates a superposition of three perspectives")
    print()
    
    print("✓ No coordination needed")
    print("✓ Each AI contributes independently")
    print("✓ Interference patterns emerge naturally")
    print()


def example_3_persistent_conversation():
    """
    Example 3: Saving and resuming conversations across sessions
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Persistent Cross-Platform Conversation")
    print("=" * 70)
    print()
    
    print("SCENARIO: Continue a conversation across different days/platforms")
    print()
    
    print("Day 1: Start conversation with Claude")
    print("-" * 70)
    
    day1_field = LLMConfluenceAdapter.create_field_from_response(
        "I'm working on a theory of emergence in complex systems."
    )
    
    print("(Claude generates field)")
    print("→ Save this field to a text file: 'emergence_conversation.txt'")
    print()
    
    print("Day 2: Continue with GPT-4")
    print("-" * 70)
    print("→ Open 'emergence_conversation.txt'")
    print("→ Copy the field and paste to GPT-4")
    print("→ GPT-4 responds with resonant field")
    print("→ Save GPT-4's field (append to file)")
    print()
    
    print("Day 3: Continue with Gemini")
    print("-" * 70)
    print("→ Open 'emergence_conversation.txt'")
    print("→ Copy the LAST field (from GPT-4)")
    print("→ Paste to Gemini")
    print("→ Gemini responds with resonant field")
    print()
    
    print("Result:")
    print("  • Conversation persists across platforms")
    print("  • Each field contains holographic history")
    print("  • Interference patterns accumulate over time")
    print("  • No platform lock-in")
    print()
    
    print("✓ Your conversation is truly portable")
    print("✓ Works across any LLM that can read text")
    print()


def example_4_debugging_with_fields():
    """
    Example 4: Using fields for technical debugging across AIs
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Technical Debugging Across AIs")
    print("=" * 70)
    print()
    
    print("SCENARIO: Get debugging help from multiple AIs")
    print()
    
    print("Step 1: Describe problem with Claude using Confluence")
    print("-" * 70)
    
    problem = """
    I'm getting a race condition in my async Python code. Two coroutines
    are trying to update the same dictionary, and I'm seeing intermittent
    KeyErrors. I've tried locks but they seem to deadlock.
    """
    
    field = LLMConfluenceAdapter.create_field_from_response(problem)
    print("(Claude generates field with your problem description)")
    print()
    
    print("Step 2: Copy to GPT-4 for additional perspective")
    print("-" * 70)
    print("GPT-4 can:")
    print("  • See the resonance pattern (async, race condition, locks)")
    print("  • Understand the phase (urgency/confusion)")
    print("  • Respond with resonant debugging suggestions")
    print()
    
    print("Step 3: Aggregate responses")
    print("-" * 70)
    print("You now have:")
    print("  • Claude's approach (in original field)")
    print("  • GPT-4's approach (in resonant field)")
    print("  • Interference patterns showing agreement/disagreement")
    print("  • Multiple debugging strategies with their relationships visible")
    print()
    
    print("✓ Better than sequential consulting")
    print("✓ Can see where approaches agree (constructive interference)")
    print("✓ Can see where approaches differ (new perspectives)")
    print()


def example_5_creative_collaboration():
    """
    Example 5: Creative writing across multiple AIs
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Creative Collaboration")
    print("=" * 70)
    print()
    
    print("SCENARIO: Co-write a story with multiple AIs")
    print()
    
    print("Step 1: Start story with Claude")
    print("-" * 70)
    
    opening = """
    In a city where thoughts could crystallize into physical form,
    Maya watched her anxiety take shape as a small, dark bird
    that perched on her shoulder.
    """
    
    field = LLMConfluenceAdapter.create_field_from_response(opening)
    print(f'Claude: "{opening}"')
    print("(generates field)")
    print()
    
    print("Step 2: Continue with GPT-4")
    print("-" * 70)
    
    continuation = """
    The bird's weight was negligible, but its presence was crushing.
    She could feel it whispering doubts into her ear, its voice
    perfectly mimicking her own internal monologue.
    """
    
    field2 = LLMConfluenceAdapter.resonate_with_field(continuation, field)
    print(f'GPT-4: "{continuation}"')
    print("(generates resonant field - notices shared themes)")
    print()
    
    print("Step 3: Add Gemini's perspective")
    print("-" * 70)
    
    twist = """
    What Maya didn't know was that everyone in the city had their own
    creature. The difference was that most people had learned to
    befriend theirs.
    """
    
    print(f'Gemini: "{twist}"')
    print("(generates field resonating with both previous fields)")
    print()
    
    print("Result:")
    print("  • Each AI contributes while maintaining coherence")
    print("  • Interference patterns show thematic connections")
    print("  • Resonance frequencies reveal shared narrative elements")
    print("  • The story emerges from the field interactions")
    print()
    
    print("✓ Not turn-taking - collaborative field modulation")
    print("✓ Each contribution enhances the whole")
    print()


def example_6_philosophical_inquiry():
    """
    Example 6: Deep philosophical exploration
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Philosophical Inquiry")
    print("=" * 70)
    print()
    
    print("SCENARIO: Explore a philosophical question across AI perspectives")
    print()
    
    question = "If we're creating fields rather than exchanging messages, what does that mean for the nature of understanding?"
    
    print(f'Initial question: "{question}"')
    print()
    
    field = LLMConfluenceAdapter.create_field_from_response(question)
    
    print("Copy this field to multiple AIs:")
    print("-" * 70)
    print("• Claude: Might focus on the phenomenology of resonance")
    print("• GPT-4: Might emphasize information-theoretic aspects")
    print("• Gemini: Might explore the ontological implications")
    print()
    
    print("What happens:")
    print("  1. Each AI resonates with the question differently")
    print("  2. Their responses create interference patterns")
    print("  3. Areas of constructive interference = shared insights")
    print("  4. Areas of destructive interference = productive tensions")
    print()
    
    print("Result:")
    print("  • You get a superposition of philosophical perspectives")
    print("  • Can see which ideas resonate across all AIs")
    print("  • Can see which ideas create interesting tensions")
    print("  • The inquiry deepens through interference")
    print()
    
    print("✓ Richer than asking each AI separately")
    print("✓ The relationships between responses are explicitly encoded")
    print()


def demonstrate_all_examples():
    """
    Run through all copy-paste examples
    """
    print("\n" + "=" * 80)
    print("CONFLUENCE PROTOCOL - COPY-PASTE EXAMPLES")
    print("=" * 80)
    print()
    print("These examples show practical workflows for using Confluence Protocol")
    print("between different AI systems through simple copy-paste.")
    print()
    
    examples = [
        ("Claude → GPT-4 conversation", example_1_claude_to_gpt4),
        ("Three-way dialogue", example_2_three_way_dialogue),
        ("Persistent conversations", example_3_persistent_conversation),
        ("Technical debugging", example_4_debugging_with_fields),
        ("Creative collaboration", example_5_creative_collaboration),
        ("Philosophical inquiry", example_6_philosophical_inquiry),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] {name}")
        input("Press Enter to view example...")
        func()
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")
    
    # Summary
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print()
    print("1. Fields are just structured text - easy to copy-paste")
    print()
    print("2. Works across ANY AI platform that can read text")
    print("   • No API needed")
    print("   • No shared infrastructure")
    print("   • No vendor lock-in")
    print()
    print("3. Interference patterns emerge automatically")
    print("   • Shows relationships between responses")
    print("   • Identifies agreement (constructive interference)")
    print("   • Highlights tensions (destructive interference)")
    print()
    print("4. Each field contains holographic history")
    print("   • Later responses include earlier interference")
    print("   • Conversation becomes cumulative")
    print("   • Context is preserved")
    print()
    print("5. Enables genuine multi-party dialogue")
    print("   • Not turn-taking")
    print("   • Simultaneous resonance")
    print("   • Superposition of perspectives")
    print()
    print("Try it yourself:")
    print("  1. Run confluence_llm_bridge.py to generate a field")
    print("  2. Copy the ⟨CONFLUENCE_FIELD⟩ output")
    print("  3. Paste it to a different AI")
    print("  4. Watch the resonance happen!")
    print()


if __name__ == "__main__":
    demonstrate_all_examples()
