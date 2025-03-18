import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
@st.cache_data
def load_data(file_path):
    try:
        xlsb_file = pd.ExcelFile(file_path, engine="pyxlsb")
        df = xlsb_file.parse(xlsb_file.sheet_names[0])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

# Currency conversion rates to INR
def get_currency_conversion_rates():
    return {
        'INR': 1.0,      
        'USD': 87.0,     
        'EUR': 94.7     
    }

# Process data
def preprocess_data(df):
    # Check if required columns exist
    required_columns = ['Short Text', 'Vendor/supplying plant', 'Pur. Doc.', 'Item', 'Plnt', 
                        '       Quantity', '      Net price', '      Net Value', 'Quantity', 
                        'Release', 'Due Days', 'Deliv.Date', 'Budget Number', 'Purch.Req.','Crcy', 'OUn']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return pd.DataFrame()
        
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df[required_columns].copy()
    df.columns = ['Short Text', 'Vendor/Supplying Plant', 'Purchase Doc', 'Item', 'Plant', 'Quantity', 
                  'Net Price', 'Net Value', 'Quantity 2', 'Release', 'Due Days', 'Delivery Date', 
                  'Budget Number', 'Purchase Req', 'Currency', 'OUn']
    
    # Handle NaN values properly
    df.dropna(subset=['Short Text'], inplace=True)
    df['Short Text'] = df['Short Text'].astype(str).str.strip().str.upper()
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['Quantity', 'Net Price', 'Net Value', 'Due Days']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert delivery date to datetime with the correct format
    df['Delivery Date'] = pd.to_datetime(df['Delivery Date'], format='%d.%m.%Y', errors='coerce')
    
    # Handle currency
    df['Currency'] = df['Currency'].astype(str).str.strip().str.upper()
    # Replace missing currency with INR
    df['Currency'] = df['Currency'].replace('', 'INR').fillna('INR')
    
    # Convert all prices to INR
    conversion_rates = get_currency_conversion_rates()
    df['Currency Rate'] = df['Currency'].map(conversion_rates)
    df['Currency Rate'] = df['Currency Rate'].fillna(1.0)  # Default to 1.0 if currency not found
    
    # Convert Net Price and Net Value to INR
    df['Net Price INR'] = df['Net Price'] * df['Currency Rate']
    df['Net Value INR'] = df['Net Value'] * df['Currency Rate']
    
    # Keep original currency for reference
    df['Original Currency'] = df['Currency']
    df['Original Net Price'] = df['Net Price']
    df['Original Net Value'] = df['Net Value']
    
    return df

# Find similar products using TF-IDF + Cosine Similarity
def find_similar_products(df, query, top_n=5):
    """
    Find products similar to the user's query using TF-IDF + Cosine Similarity.
    """
    # Get unique product descriptions
    product_descriptions = df['Short Text'].unique()
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the product descriptions
    tfidf_matrix = vectorizer.fit_transform(product_descriptions)
    
    # Transform the query
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and product descriptions
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get the indices of the top N most similar products
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Return the matched products and their similarity scores
    matches = [(product_descriptions[i], cosine_similarities[i]) for i in top_indices]
    
    return matches

# Find and analyze suppliers with detailed comparison
def find_and_analyze_suppliers(df, query, top_n=5):
    query = query.strip().upper()
    
    # Try exact match first
    matched_df = df[df['Short Text'] == query].copy()
    
    # If no exact match, try partial match
    if matched_df.empty:
        matched_df = df[df['Short Text'].str.contains(query, na=False)].copy()
        if not matched_df.empty:
            st.info(f"No exact match found for '{query}'. Showing partial matches.")
    
    if matched_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Store raw transaction data for display
    all_transactions = matched_df.copy()
    
    # Aggregate by supplier
    aggregated = []
    
    # Group by vendor
    vendor_groups = matched_df.groupby('Vendor/Supplying Plant')
    
    for vendor, group in vendor_groups:
        # Basic aggregation
        total_quantity = group['Quantity'].sum()
        total_value_inr = group['Net Value INR'].sum()
        avg_price_inr = group['Net Price INR'].mean()
        avg_due_days = group['Due Days'].mean()
        order_count = len(group)
        
        # Calculate original currency values
        original_currency = group['Original Currency'].mode().iloc[0]  # Most frequent currency
        if original_currency == 'INR':
            avg_price_original = avg_price_inr
            total_value_original = total_value_inr
        else:
            # Convert INR values back to the original currency
            conversion_rate = group['Currency Rate'].mean()
            avg_price_original = avg_price_inr / conversion_rate if conversion_rate > 0 else 0
            total_value_original = total_value_inr / conversion_rate if conversion_rate > 0 else 0
        
        # Get the most common measurement unit
        OUn = group['OUn'].mode().iloc[0] if not group['OUn'].mode().empty else ''
        
        # Calculate price consistency
        price_std = group['Net Price INR'].std()
        price_consistency = 1 - (price_std / avg_price_inr if avg_price_inr > 0 else 0)
        price_consistency = max(0, min(1, price_consistency))  # Clamp between 0 and 1
        
        # Identify most recent purchase
        most_recent_date = group['Delivery Date'].max()
        
        # Collect detailed purchase history
        purchase_docs = ', '.join(group['Purchase Doc'].astype(str).unique())
        delivery_dates = group['Delivery Date'].dropna()
        delivery_dates_str = ', '.join([d.strftime('%Y-%m-%d') for d in delivery_dates if pd.notna(d)])
        
        # Get currencies used
        currencies = group['Original Currency'].unique()
        currencies_str = ', '.join(currencies)
        
        # Calculate price trend (if multiple purchases)
        price_trend = "N/A"
        if len(group) > 1 and pd.notna(most_recent_date):
            # Sort by date
            sorted_group = group.sort_values('Delivery Date')
            first_price = sorted_group['Net Price INR'].iloc[0]
            last_price = sorted_group['Net Price INR'].iloc[-1]
            
            if first_price == 0:
                price_trend = "N/A"
            elif abs(last_price - first_price) / first_price < 0.01:  # Less than 1% change
                price_trend = "Stable"
            elif last_price > first_price:
                price_trend = "Increasing"
            else:
                price_trend = "Decreasing"
        
        # Store in aggregated results
        aggregated.append({
            'Vendor/Supplying Plant': vendor,
            'Purchase Doc': purchase_docs,
            'Plant': group['Plant'].iloc[0],
            'Total Quantity': total_quantity,
            'OUn': OUn,  # Include measurement unit
            'Avg Price (INR)': avg_price_inr,
            'Avg Price (Original Currency)': avg_price_original,
            'Total Value (INR)': total_value_inr,
            'Total Value (Original Currency)': total_value_original,
            'Avg Due Days': avg_due_days,
            'Order Count': order_count,
            'Price Consistency': price_consistency * 100,  # Convert to percentage
            'Price Trend': price_trend,
            'Delivery Date': delivery_dates_str,  # Include delivery dates
            #'Currencies Used': currencies_str,
            'Original Currency': original_currency,  # Add original currency
            'Most Recent Date': most_recent_date  # Add most recent date for scoring
        })
    
    # Convert to DataFrame
    agg_df = pd.DataFrame(aggregated)
    
    # Calculate scores for comparison across vendors
    if not agg_df.empty and len(agg_df) > 0:
        # Base metrics normalization
        quantity_max = agg_df['Total Quantity'].max() if not agg_df['Total Quantity'].empty else 0
        price_min = agg_df['Avg Price (INR)'].min() if not agg_df['Avg Price (INR)'].empty else 0  # Lower price is better
        due_days_min = agg_df['Avg Due Days'].min() if not agg_df['Avg Due Days'].empty else 0  # Lower days is better
        
        # Avoid division by zero
        agg_df['Quantity Score'] = agg_df['Total Quantity'] / quantity_max if quantity_max > 0 else 0
        
        # Price score - lower is better
        if price_min > 0:
            agg_df['Price Score'] = price_min / agg_df['Avg Price (INR)']
        else:
            agg_df['Price Score'] = 0
            
        # Due days score - lower is better
        if due_days_min > 0:
            agg_df['Due Days Score'] = due_days_min / agg_df['Avg Due Days']
        else:
            agg_df['Due Days Score'] = 1  # Perfect score if zero days
        
        # Special considerations for unbalanced purchase history
        
        # 1. Recency bonus - more recent suppliers get a bonus
        if 'Most Recent Date' in agg_df.columns and agg_df['Most Recent Date'].notna().any():
            latest_date = agg_df['Most Recent Date'].max()
            earliest_date = agg_df['Most Recent Date'].min()
            
            if pd.notna(latest_date) and pd.notna(earliest_date) and latest_date != earliest_date:
                date_range = (latest_date - earliest_date).total_seconds()
                if date_range > 0:
                    agg_df['Recency Score'] = agg_df.apply(
                        lambda row: (row['Most Recent Date'] - earliest_date).total_seconds() / date_range if pd.notna(row['Most Recent Date']) else 0,
                        axis=1
                    )
                else:
                    agg_df['Recency Score'] = 1.0
            else:
                agg_df['Recency Score'] = 1.0
        else:
            agg_df['Recency Score'] = 0.5  # Neutral score if no date
            
        # 2. Experience factor - suppliers with more orders get a small advantage, but not overwhelming
        # We use a logarithmic scale to prevent suppliers with many small orders from dominating
        if agg_df['Order Count'].max() > 0:
            agg_df['Experience Score'] = agg_df['Order Count'].apply(lambda x: np.log1p(x) / np.log1p(agg_df['Order Count'].max()))
        else:
            agg_df['Experience Score'] = 0
            
        # 3. Consistency bonus - more consistent pricing gets a bonus
        agg_df['Consistency Score'] = agg_df['Price Consistency'] / 100  # Convert back to 0-1 scale
        
        # Calculate weighted final score
        # For unbalanced purchase history, we reduce the weight of experience
        # and increase the importance of price and consistency
        agg_df['Score(%)'] = ((agg_df['Quantity Score'] * 0.25 +
                          agg_df['Price Score'] * 0.30 +
                          agg_df['Due Days Score'] * 0.15 +
                          agg_df['Recency Score'] * 0.10 +
                          agg_df['Experience Score'] * 0.10 +  # Lower weight for experience
                          agg_df['Consistency Score'] * 0.10) * 100).clip(0, 100)
        
        # Round score to 2 decimal places
        agg_df['Score(%)'] = agg_df['Score(%)'].round(2)
    
    # Return top n suppliers by score
    if not agg_df.empty and 'Score(%)' in agg_df.columns:
        return agg_df.nlargest(top_n, 'Score(%)'), all_transactions
    else:
        return agg_df, all_transactions

def generate_supplier_insights(supplier_data, all_suppliers_data):
    """Generate insights about a specific supplier compared to others"""
    insights = {}
    
    # Extract data
    supplier_name = supplier_data['Vendor/Supplying Plant']
    avg_price = supplier_data['Avg Price (INR)']
    due_days = supplier_data['Avg Due Days']
    price_consistency = supplier_data['Price Consistency']
    order_count = supplier_data['Order Count']
    
    # Calculate averages across all suppliers
    avg_price_all = all_suppliers_data['Avg Price (INR)'].mean()
    avg_due_days_all = all_suppliers_data['Avg Due Days'].mean()
    avg_consistency_all = all_suppliers_data['Price Consistency'].mean()
    
    # Price insight
    price_diff_pct = ((avg_price / avg_price_all) - 1) * 100 if avg_price_all > 0 else 0
    if abs(price_diff_pct) < 3:
        insights['price'] = f"This supplier offers market competitive pricing at around {avg_price:.2f} INR, which is close to the average price of {avg_price_all:.2f} INR among all suppliers."
    elif price_diff_pct < 0:
        insights['price'] = f"This supplier offers better pricing at {avg_price:.2f} INR, which is {abs(price_diff_pct):.1f}% lower than the average price of {avg_price_all:.2f} INR among all suppliers."
    else:
        insights['price'] = f"This supplier charges a premium price of {avg_price:.2f} INR, which is {price_diff_pct:.1f}% higher than the average price of {avg_price_all:.2f} INR among all suppliers."
    
    # Delivery time insight
    days_diff_pct = ((due_days / avg_due_days_all) - 1) * 100 if avg_due_days_all > 0 else 0
    if abs(days_diff_pct) < 5:
        insights['delivery'] = f"This supplier delivers in about {due_days:.1f} days, which is comparable to the average delivery time of {avg_due_days_all:.1f} days."
    elif days_diff_pct < 0:
        insights['delivery'] = f"This supplier is faster than average, delivering in {due_days:.1f} days compared to the average of {avg_due_days_all:.1f} days ({abs(days_diff_pct):.1f}% faster)."
    else:
        insights['delivery'] = f"This supplier takes longer to deliver ({due_days:.1f} days) compared to the average of {avg_due_days_all:.1f} days ({days_diff_pct:.1f}% longer)."
    
    # Consistency insight
    if price_consistency >= 90:
        insights['consistency'] = f"This supplier has excellent price consistency at {price_consistency:.1f}%, indicating very reliable and predictable pricing."
    elif price_consistency >= 75:
        insights['consistency'] = f"This supplier has good price consistency at {price_consistency:.1f}%, showing generally stable pricing."
    elif price_consistency >= 60:
        insights['consistency'] = f"This supplier has moderate price consistency at {price_consistency:.1f}%, with some price fluctuations."
    else:
        insights['consistency'] = f"This supplier has low price consistency at {price_consistency:.1f}%, indicating variable pricing between orders."
    
    # Experience insight
    if order_count > 5:
        insights['experience'] = f"You have extensive experience with this supplier ({order_count} previous orders), which reduces risk and uncertainty."
    elif order_count > 2:
        insights['experience'] = f"You have moderate experience with this supplier ({order_count} previous orders), providing some confidence in their performance."
    else:
        insights['experience'] = f"You have limited experience with this supplier (only {order_count} previous orders), which may present some uncertainty."
    
    # Overall recommendation
    top_supplier = all_suppliers_data.iloc[0]['Vendor/Supplying Plant']
    if supplier_name == top_supplier:
        insights['recommendation'] = "This is the top-recommended supplier based on our comprehensive analysis of price, delivery time, consistency and past experience."
    else:
        supplier_rank = all_suppliers_data[all_suppliers_data['Vendor/Supplying Plant'] == supplier_name].index[0] + 1
        insights['recommendation'] = f"This supplier ranks #{supplier_rank} in our analysis. While not the top recommendation, they may have specific strengths that suit your current needs."
    
    return insights

# Streamlit UI
st.title("Supplier Recommendation Agent")

# Hardcoded file path - replace with your actual path
file_path = r"ME2N jan_dec.xlsb"  

with st.spinner("Loading data..."):
    df = load_data(file_path)
    if not df.empty:
        df = preprocess_data(df)
        st.success(f"Data loaded successfully with {len(df)} records.")
    else:
        st.error(f"Failed to load data from {file_path}. Please check if the file exists and has the correct format.")

# Display currency conversion rates
with st.expander("Currency Conversion Rates"):
    rates = get_currency_conversion_rates()
    st.write("All prices are converted to INR using the following rates:")
    for currency, rate in rates.items():
        if currency != 'INR':
            st.write(f"1 {currency} = {rate} INR")

query = st.text_input("Enter Product Short Text:")
if query:
    if not df.empty:
        # Find similar products
        similar_products = find_similar_products(df, query, top_n=5)
        
        if similar_products:
            st.subheader(f"Similar Products for '{query}'")
            st.write("We found the following similar products. Please select one:")
            
            # Display similar products with their similarity scores
            selected_product = st.selectbox(
                "Select a product:",
                options=[f"{product} (Score: {score:.2f})" for product, score in similar_products]
            )
            
            # Extract the selected product name
            selected_product = selected_product.split(" (Score:")[0]
            
            # Proceed with supplier analysis for the selected product
            with st.spinner(f"Analyzing suppliers for '{selected_product}'..."):
                aggregated_results, all_transactions = find_and_analyze_suppliers(df, selected_product)
                
            if not aggregated_results.empty:
                st.subheader(f"Supplier Analysis for '{selected_product}'")
                
                # Display the aggregated results in a table
                display_cols = ['Vendor/Supplying Plant', 'Total Quantity', 'OUn', 'Avg Price (INR)', 'Avg Price (Original Currency)',
                               'Total Value (INR)', 'Total Value (Original Currency)','Original Currency', 'Avg Due Days', 'Order Count', 
                               'Price Consistency', 'Score(%)']
                st.dataframe(aggregated_results[display_cols])
                
                # Check for unbalanced purchase history
                order_counts = aggregated_results['Order Count'].tolist()
                has_unbalanced_history = (max(order_counts) - min(order_counts) >= 2) if len(order_counts) > 1 else False
                
                if has_unbalanced_history:
                    st.info("Note: Unbalanced purchase history detected. Score calculation has been adjusted accordingly.")
                
                # Show all individual transactions
                with st.expander("View All Transactions"):
                    if not all_transactions.empty:
                        display_transaction_cols = ['Vendor/Supplying Plant', 'Purchase Doc', 'Quantity', 'OUn',
                                                   'Original Currency', 'Original Net Price', 'Original Net Value',
                                                   'Net Price INR', 'Net Value INR', 'Due Days', 'Delivery Date']
                        st.dataframe(all_transactions[display_transaction_cols])
                
                # New section - Allow user to select a supplier
                st.subheader("Select a Supplier for Detailed Analysis")
                
                # Get list of suppliers
                supplier_list = aggregated_results['Vendor/Supplying Plant'].tolist()
                
                # Allow user to select a supplier
                selected_supplier = st.selectbox(
                    "Choose a supplier to view detailed insights:",
                    options=supplier_list,
                    index=0  # Default to the top supplier
                )
                
                # Get the data for the selected supplier
                selected_supplier_data = aggregated_results[aggregated_results['Vendor/Supplying Plant'] == selected_supplier].iloc[0]
                
                # Generate insights about the selected supplier
                insights = generate_supplier_insights(selected_supplier_data, aggregated_results)
                
                # Display supplier insights in a nicely formatted card
                st.subheader(f"Insights for {selected_supplier}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Pricing")
                    st.markdown(insights['price'])
                    
                    st.markdown("### Delivery Time")
                    st.markdown(insights['delivery'])
                
                with col2:
                    st.markdown("### Price Consistency")
                    st.markdown(insights['consistency'])
                    
                    st.markdown("### Experience")
                    st.markdown(insights['experience'])
                
                st.markdown("### Recommendation")
                st.markdown(f"**{insights['recommendation']}**")
                
                # Additional metrics in expander
                with st.expander("View Detailed Metrics"):
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric(
                            "Average Price (INR)", 
                            f"{selected_supplier_data['Avg Price (INR)']:.2f}"
                        )
                    
                    with metric_cols[1]:
                        st.metric(
                            "Due days", 
                            f"{selected_supplier_data['Avg Due Days']:.1f}"
                        )

                    with metric_cols[2]:
                        st.metric(
                            "Total Quantity", 
                            f"{selected_supplier_data['Total Quantity']:.1f}"
                        )
                    
                    with metric_cols[3]:
                        st.metric(
                            "Price Consistency", 
                            f"{selected_supplier_data['Price Consistency']:.1f}%"
                        )
                    
                    with metric_cols[0]:
                        st.metric(
                            "Total Orders", 
                            f"{selected_supplier_data['Order Count']:.2f}"
                        )
                
                # Show JSON output for developers
                with st.expander("JSON Output"):
                    st.subheader("JSON Data")
                    output_json = aggregated_results.to_dict(orient='records')
                    st.json(output_json)
            else:
                st.warning(f"No matches found for '{selected_product}'.")
        else:
            st.warning(f"No similar products found for '{query}'.")
    else:
        st.error("Error loading or processing the data file. Please check if the file exists and has the correct format.")
