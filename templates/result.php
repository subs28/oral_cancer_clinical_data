<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oral Cancer Prediction Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: rgba(17, 219, 207, 0.8);
            color: rgba(0, 0, 0, 0.6);
            margin: 0;
            padding: 0;
        }
        h1, h2, p {
            margin: 10px 0;
        }
        h1 {
            text-align: center;
        }
        p {
            font-size: 18px;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        li {
            margin-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            border: 1px solid #ddd;
            margin-top: 20px;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <h1>Oral Cancer Prediction Result</h1>
    <h2>Prediction for Clinical Data</h2>
    {% if result_cl == 0 %}
        <p>Risk of cancer is very low. Need not worry, relax.</p>
    {% elif result_cl == 1 %}
        <p>Risk of cancer is a little high. Consult an oncologist immediately. Alert!</p>
    {% else %}
        <p>Invalid result. Please check the prediction again.</p>
    {% endif %}
    <div>{% if result_cl == 1  %}
    <h2>Oncology Centers near you : </h2>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Address</th>
                    <th>Contact</th>
                </tr>
            </thead>
            <tbody>
                {% for hospital in hospitals_list %}
                    <tr>
                        <td>{{ hospital.Name }}</td>
                        <td>{{ hospital.Address }}</td>
                        <td><a href="tel:{{ hospital.Contact }}">{{ hospital.Contact }}</a></td>
                    </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>


</body>
</html>
