# Generated by Django 4.0 on 2021-12-27 19:40

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
        ('application', '0010_alter_userresponse_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dept_userresponse',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='auth.user'),
        ),
    ]
