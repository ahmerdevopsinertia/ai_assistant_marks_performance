import { Controller, Get, Post, Body } from '@nestjs/common';
import { AppService } from './app.service';
import { AnalysisService } from './analysis.service';

@Controller()
export class AppController {
  constructor(
    private readonly appService: AppService
  ) { }

  @Get()
  getHello(): string {
    return this.appService.getHello();
  }

  @Post('/analysis/yearly')
  async analyzeYear(@Body() { year }: { year: number }) {
    return this.appService.analyzeAcademicYear(year);
  }

  @Post('/analysis/yearly/enhanced')
  async analyzeYearEnhanced(@Body() { year }: { year: number }) {
    return this.appService.analyzeAcademicYearEnhanced(year);
  }

  @Post('/analysis/yearly/kmeans')
  async analyzeYearKmeans(@Body() { year }: { year: number }) {
    return this.appService.analyzeAcademicYearKmeans(year);
  }
}
