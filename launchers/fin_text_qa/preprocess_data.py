"""
Preprocess financial QA data with different input combinations.

Generates separate parquet files for each combination:
- question_only: Company info + question
- question_fundamental: + macro + fundamental
- question_news: + news
- question_full: + close prices (compact format)

Usage:
    python preprocess_data.py --input_dir /path/to/input --output_dir /path/to/output
    python preprocess_data.py --input_dir /path/to/input --output_dir /path/to/output --combinations question_only question_full
"""

import argparse
import os
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class PromptSections:
    """Parsed sections from the original prompt."""
    company_info: str = ""
    price_data: str = ""
    close_prices: List[float] = None
    macro_data: str = ""
    fundamental_data: str = ""
    news_data: str = ""
    question: str = ""

    def __post_init__(self):
        if self.close_prices is None:
            self.close_prices = []


def parse_prompt_sections(prompt_content: str) -> PromptSections:
    """Parse the original prompt into separate sections."""
    sections = PromptSections()

    # Define section markers
    markers = [
        ("=== Company Information ===", "company_info"),
        ("=== Historical Price Data", "price_data"),
        ("=== Historical Close Prices", "price_data"),  # Handle already converted
        ("=== Macroeconomic Indicators ===", "macro_data"),
        ("=== Fundamental Data ===", "fundamental_data"),
        ("=== Recent News ===", "news_data"),
        ("=== News ===", "news_data"),
        ("=== Question ===", "question"),
    ]

    # Find all section positions
    section_positions = []
    for marker, section_name in markers:
        pos = prompt_content.find(marker)
        if pos != -1:
            section_positions.append((pos, marker, section_name))

    # Sort by position
    section_positions.sort(key=lambda x: x[0])

    # Extract each section
    for i, (pos, marker, section_name) in enumerate(section_positions):
        # Find end of this section (start of next section or end of string)
        if i + 1 < len(section_positions):
            end_pos = section_positions[i + 1][0]
        else:
            end_pos = len(prompt_content)

        # Extract content (without the marker itself)
        content = prompt_content[pos + len(marker):end_pos].strip()

        # Store in appropriate field
        if section_name == "company_info":
            sections.company_info = content
        elif section_name == "price_data":
            sections.price_data = content
            # Also extract close prices
            sections.close_prices = extract_close_prices(content)
        elif section_name == "macro_data":
            sections.macro_data = content
        elif section_name == "fundamental_data":
            sections.fundamental_data = content
        elif section_name == "news_data":
            sections.news_data = content
        elif section_name == "question":
            sections.question = content

    return sections


def extract_close_prices(price_section: str) -> List[float]:
    """Extract close prices from the price section."""
    close_prices = []

    # Check if it's already in compact format
    if "Close prices:" in price_section:
        # Extract from compact format: "Close prices: 195.39, 203.64, ..."
        match = re.search(r'Close prices:\s*([\d.,\s]+)', price_section)
        if match:
            prices_str = match.group(1)
            for p in prices_str.split(','):
                try:
                    close_prices.append(float(p.strip()))
                except ValueError:
                    pass
        return close_prices

    # Parse CSV format
    lines = price_section.strip().split('\n')
    for line in lines:
        if line.startswith('Date,') or not line.strip():
            continue

        parts = line.split(',')
        if len(parts) >= 5:
            try:
                close_price = float(parts[4])  # Close is 5th column
                close_prices.append(close_price)
            except (ValueError, IndexError):
                pass

    return close_prices


def build_prompt_for_combination(sections: PromptSections, combination: str) -> str:
    """Build prompt for a specific combination."""
    parts = []

    # Always include company info
    if sections.company_info:
        parts.append("=== Company Information ===")
        parts.append(sections.company_info)
        parts.append("")

    if combination == "question_only":
        # Only company info + question
        pass

    elif combination == "question_fundamental":
        # Add macro and fundamental
        if sections.macro_data:
            parts.append("=== Macroeconomic Indicators ===")
            parts.append(sections.macro_data)
            parts.append("")

        if sections.fundamental_data:
            parts.append("=== Fundamental Data ===")
            parts.append(sections.fundamental_data)
            parts.append("")

    elif combination == "question_news":
        # Add macro, fundamental, and news
        if sections.macro_data:
            parts.append("=== Macroeconomic Indicators ===")
            parts.append(sections.macro_data)
            parts.append("")

        if sections.fundamental_data:
            parts.append("=== Fundamental Data ===")
            parts.append(sections.fundamental_data)
            parts.append("")

        parts.append("=== Recent News ===")
        if sections.news_data and sections.news_data.strip():
            parts.append(sections.news_data)
        else:
            parts.append("No recent news available.")
        parts.append("")

    elif combination == "question_full":
        # Add everything including compact close prices
        if sections.macro_data:
            parts.append("=== Macroeconomic Indicators ===")
            parts.append(sections.macro_data)
            parts.append("")

        if sections.fundamental_data:
            parts.append("=== Fundamental Data ===")
            parts.append(sections.fundamental_data)
            parts.append("")

        parts.append("=== Recent News ===")
        if sections.news_data and sections.news_data.strip():
            parts.append(sections.news_data)
        else:
            parts.append("No recent news available.")
        parts.append("")

        # Add compact close prices
        if sections.close_prices:
            parts.append(f"=== Historical Close Prices ({len(sections.close_prices)} days) ===")
            prices_str = ", ".join(f"{p:.2f}" for p in sections.close_prices)
            parts.append(f"Close prices: {prices_str}")
            parts.append("")

    # Always add question at the end
    if sections.question:
        parts.append("=== Question ===")
        parts.append(sections.question)

    return "\n".join(parts)


def process_dataframe(df: pd.DataFrame, combination: str) -> pd.DataFrame:
    """Process entire dataframe for a specific combination."""
    processed_data = []

    for idx in range(len(df)):
        row = df.iloc[idx].to_dict()

        # Get original prompt content (handle numpy array wrapping)
        prompt_list = row.get('prompt', [])
        # Convert numpy array to list if needed
        if hasattr(prompt_list, 'tolist'):
            prompt_list = prompt_list.tolist()

        if isinstance(prompt_list, list) and len(prompt_list) > 0:
            original_content = prompt_list[0].get('content', '')
        else:
            original_content = ''

        # Parse sections
        sections = parse_prompt_sections(original_content)

        # Build new prompt for this combination
        new_content = build_prompt_for_combination(sections, combination)

        # Update row
        row['prompt'] = [{'role': 'user', 'content': new_content}]
        processed_data.append(row)

        if (idx + 1) % 5000 == 0:
            print(f"    Processed {idx + 1}/{len(df)} samples")

    return pd.DataFrame(processed_data)


def show_example(df: pd.DataFrame, combinations: List[str]):
    """Show example prompts for each combination."""
    sample = df.iloc[0]
    original_content = sample['prompt'][0]['content']
    sections = parse_prompt_sections(original_content)

    print("\n" + "="*80)
    print("ORIGINAL PROMPT SECTIONS:")
    print("="*80)
    print(f"Company Info: {len(sections.company_info)} chars")
    print(f"Price Data: {len(sections.price_data)} chars ({len(sections.close_prices)} close prices)")
    print(f"Macro Data: {len(sections.macro_data)} chars")
    print(f"Fundamental Data: {len(sections.fundamental_data)} chars")
    print(f"News Data: {len(sections.news_data)} chars")
    print(f"Question: {len(sections.question)} chars")
    print(f"Total Original: {len(original_content)} chars")

    for combo in combinations:
        new_prompt = build_prompt_for_combination(sections, combo)
        print(f"\n{'='*80}")
        print(f"COMBINATION: {combo} ({len(new_prompt)} chars)")
        print("="*80)
        # Show first 1500 chars
        print(new_prompt[:1500])
        if len(new_prompt) > 1500:
            print(f"\n... [truncated, total: {len(new_prompt)} chars]")


def main():
    parser = argparse.ArgumentParser(description='Preprocess financial QA data with different combinations')
    parser.add_argument('--input_dir', type=str, default='/fsx/ubuntu/users/aosong/fin/data/finance_qa',
                        help='Input directory with train/test parquet files')
    parser.add_argument('--output_dir', type=str, default='/fsx/ubuntu/users/aosong/fin/data/finance_qa_text',
                        help='Output directory for processed files')
    parser.add_argument('--combinations', nargs='+',
                        default=['question_only', 'question_fundamental', 'question_news', 'question_full'],
                        help='Combinations to generate')
    parser.add_argument('--show_example', action='store_true', help='Show example before processing')
    parser.add_argument('--train_file', type=str, default='train_20k.parquet', help='Train file name')
    parser.add_argument('--test_file', type=str, default='test_2k.parquet', help='Test file name')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Combinations: {args.combinations}")

    # Load data
    train_path = os.path.join(args.input_dir, args.train_file)
    test_path = os.path.join(args.input_dir, args.test_file)

    print(f"\nLoading train data from {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"Loaded {len(train_df)} train samples")

    print(f"\nLoading test data from {test_path}")
    test_df = pd.read_parquet(test_path)
    print(f"Loaded {len(test_df)} test samples")

    # Show example if requested
    if args.show_example:
        show_example(train_df, args.combinations)
        print("\n")

    # Process each combination
    for combo in args.combinations:
        print(f"\n{'='*60}")
        print(f"Processing combination: {combo}")
        print("="*60)

        # Create subdirectory for this combination
        combo_dir = os.path.join(args.output_dir, combo)
        os.makedirs(combo_dir, exist_ok=True)

        # Process train
        print(f"  Processing train set...")
        train_processed = process_dataframe(train_df, combo)
        train_output = os.path.join(combo_dir, 'train.parquet')
        train_processed.to_parquet(train_output, index=False)
        print(f"  Saved to {train_output}")

        # Process test
        print(f"  Processing test set...")
        test_processed = process_dataframe(test_df, combo)
        test_output = os.path.join(combo_dir, 'test.parquet')
        test_processed.to_parquet(test_output, index=False)
        print(f"  Saved to {test_output}")

    print(f"\n{'='*60}")
    print("Done! Generated files:")
    print("="*60)
    for combo in args.combinations:
        combo_dir = os.path.join(args.output_dir, combo)
        print(f"  {combo}/")
        print(f"    - train.parquet")
        print(f"    - test.parquet")


if __name__ == '__main__':
    main()
