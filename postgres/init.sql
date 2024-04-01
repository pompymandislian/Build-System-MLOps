CREATE TABLE user_predicts(
	id SERIAL PRIMARY KEY,
	transaction numeric,
	age varchar(10),
	tenure numeric,
	num_pages_visited numeric,
	has_credit_card boolean,
	items_in_cart numeric,
	purchase_prediction boolean
);