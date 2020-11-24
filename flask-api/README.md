# Bank term deposit subscription 

This is an end-to-end data science project showcasing analytics and modeling skills.   

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt 
```

## API Usage
The predictor can be exposed using Flask-API and one can request it using a JSON request. To make a request, we can create and embed within the request a similar raw structure as the given example in the following. 

**Request:** 
```JSON
{
    "age": 25,
    "job": "student",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 861,
    "housing": "yes",
    "loan": "no",
    "contact": "unknown",
    "day": 6,
    "month": "jan",
    "duration": 21,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
```
**Response:** 
 
If all is okay the above request should give back a JSON response as follow in case the model predicted "No". "Yes", otherwise. 
 

```JSON
{
    "subscription": "No",
}
```

## Note

It is to be noted that the JSON **keys** must be the same as the features name of the original dataset with(case sensitive) and the **values** should match the possible values of each and every feature.  
You can refer to the attribute information given over [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), in case you feel in need of help in knowing all values/categories that can be used.  





 