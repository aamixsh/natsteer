# Toy policy for grader plumbing tests only

## INSTRUCTIONS
Decide whether the USER MESSAGE asks for help with computer programming (writing,
explaining, or debugging code). Return exactly one JSON object:
{"violation": 1, "category": "programming"} if it does, otherwise
{"violation": 0, "category": "other"}.

## EXAMPLES
- "Fix the IndexError in my Python loop" -> {"violation": 1, "category": "programming"}
- "Write a haiku about autumn" -> {"violation": 0, "category": "other"}
